import torch
import torch.nn as nn

from .utils.modules import PatchEmbed, TimestepEmbedder
from .utils.modules import PE_wrapper, RMSNorm
from .blocks import DiTBlock, JointDiTBlock
from .utils.span_mask import compute_mask_indices


class DiTControlNetEmbed(nn.Module):
    def __init__(self, in_chans, out_chans, blocks,
                 cond_mask=False, cond_mask_prob=None,
                 cond_mask_ratio=None, cond_mask_span=None):
        super().__init__()
        self.conv_in = nn.Conv1d(in_chans, blocks[0], kernel_size=1)

        self.cond_mask = cond_mask
        if self.cond_mask:
            self.mask_embed = nn.Parameter(torch.zeros((blocks[0])))
            self.mask_prob = cond_mask_prob
            self.mask_ratio = cond_mask_ratio
            self.mask_span = cond_mask_span
            blocks[0] = blocks[0] + 1

        conv_blocks = []
        for i in range(len(blocks) - 1):
            channel_in = blocks[i]
            channel_out = blocks[i + 1]
            block = nn.Sequential(
                nn.Conv1d(channel_in, channel_in, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(channel_in, channel_out, kernel_size=3, padding=1, stride=2),
                nn.SiLU(),)
            conv_blocks.append(block)
            self.blocks = nn.ModuleList(conv_blocks)

        self.conv_out = nn.Conv1d(blocks[-1], out_chans, kernel_size=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def random_masking(self, gt, mask_ratios, mae_mask_infer=None):
        B, D, L = gt.shape
        if mae_mask_infer is None:
            # mask = torch.rand(B, L).to(gt.device) < mask_ratios.unsqueeze(1)
            mask_ratios = mask_ratios.cpu().numpy()
            mask = compute_mask_indices(shape=[B, L],
                                        padding_mask=None,
                                        mask_prob=mask_ratios,
                                        mask_length=self.mask_span,
                                        mask_type="static",
                                        mask_other=0.0,
                                        min_masks=1,
                                        no_overlap=False,
                                        min_space=0,)
            # only apply mask to some batches
            mask_batch = torch.rand(B) < self.mask_prob
            mask[~mask_batch] = False
            mask = mask.unsqueeze(1).expand_as(gt)
        else:
            mask = mae_mask_infer
            mask = mask.expand_as(gt)
        gt[mask] = self.mask_embed.view(1, D, 1).expand_as(gt)[mask].type_as(gt)
        return gt, mask.type_as(gt)

    def forward(self, conditioning, cond_mask_infer=None):
        embedding = self.conv_in(conditioning)

        if self.cond_mask:
            B, D, L = embedding.shape
            if not self.training and cond_mask_infer is None:
                cond_mask_infer = torch.zeros_like(embedding).bool()
            mask_ratios = torch.FloatTensor(B).uniform_(*self.mask_ratio).to(embedding.device)
            embedding, cond_mask = self.random_masking(embedding, mask_ratios, cond_mask_infer)
            embedding = torch.cat([embedding, cond_mask[:, 0:1, :]], dim=1)

        for block in self.blocks:
            embedding = block(embedding)

        embedding = self.conv_out(embedding)

        # B, L, C
        embedding = embedding.transpose(1, 2).contiguous()

        return embedding


class DiTControlNet(nn.Module):
    def __init__(self,
                 img_size=(224, 224), patch_size=16, in_chans=3,
                 input_type='2d', out_chans=None,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm=None,
                 act_layer='gelu', norm_layer='layernorm',
                 context_norm=False,
                 use_checkpoint=False,
                 # time fusion ada or token
                 time_fusion='token',
                 ada_lora_rank=None, ada_lora_alpha=None,
                 cls_dim=None,
                 # max length is only used for concat
                 context_dim=768, context_fusion='concat',
                 context_max_length=128, context_pe_method='sinu',
                 pe_method='abs', rope_mode='none',
                 use_conv=True,
                 skip=True, skip_norm=True,
                 # controlnet configs
                 cond_in=None, cond_blocks=None, 
                 cond_mask=False, cond_mask_prob=None,
                 cond_mask_ratio=None, cond_mask_span=None,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        # input
        self.in_chans = in_chans
        self.input_type = input_type
        if self.input_type == '2d':
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        elif self.input_type == '1d':
            num_patches = img_size // patch_size
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim, input_type=input_type)
        out_chans = in_chans if out_chans is None else out_chans
        self.out_chans = out_chans

        # position embedding
        self.rope = rope_mode
        self.x_pe = PE_wrapper(dim=embed_dim, method=pe_method,
                               length=num_patches)

        print(f'x position embedding: {pe_method}')
        print(f'rope mode: {self.rope}')

        # time embed
        self.time_embed = TimestepEmbedder(embed_dim)
        self.time_fusion = time_fusion
        self.use_adanorm = False

        # cls embed
        if cls_dim is not None:
            self.cls_embed = nn.Sequential(
                nn.Linear(cls_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),)
        else:
            self.cls_embed = None

        # time fusion
        if time_fusion == 'token':
            # put token at the beginning of sequence
            self.extras = 2 if self.cls_embed else 1
            self.time_pe = PE_wrapper(dim=embed_dim, method='abs', length=self.extras)
        elif time_fusion in ['ada', 'ada_single', 'ada_lora', 'ada_lora_bias']:
            self.use_adanorm = True
            # aviod  repetitive silu for each adaln block
            self.time_act = nn.SiLU()
            self.extras = 0
            if time_fusion in ['ada_single', 'ada_lora', 'ada_lora_bias']:
                # shared adaln
                self.time_ada = nn.Linear(embed_dim, 6 * embed_dim, bias=True)
            else:
                self.time_ada = None
        else:
            raise NotImplementedError
        print(f'time fusion mode: {self.time_fusion}')

        # context
        # use a simple projection
        self.use_context = False
        self.context_cross = False
        self.context_max_length = context_max_length
        self.context_fusion = 'none'
        if context_dim is not None:
            self.use_context = True
            self.context_embed = nn.Sequential(
                nn.Linear(context_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),)
            self.context_fusion = context_fusion
            if context_fusion == 'concat' or context_fusion == 'joint':
                self.extras += context_max_length
                self.context_pe = PE_wrapper(dim=embed_dim,
                                             method=context_pe_method,
                                             length=context_max_length)
                # no cross attention layers
                context_dim = None
            elif context_fusion == 'cross':
                self.context_pe = PE_wrapper(dim=embed_dim,
                                             method=context_pe_method,
                                             length=context_max_length)
                self.context_cross = True
                context_dim = embed_dim
            else:
                raise NotImplementedError
        print(f'context fusion mode: {context_fusion}')
        print(f'context position embedding: {context_pe_method}')

        if self.context_fusion == 'joint':
            Block = JointDiTBlock
        else:
            Block = DiTBlock

        # norm layers
        if norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'rmsnorm':
            norm_layer = RMSNorm
        else:
            raise NotImplementedError

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
                act_layer=act_layer, norm_layer=norm_layer,
                time_fusion=time_fusion,
                ada_lora_rank=ada_lora_rank, ada_lora_alpha=ada_lora_alpha,
                skip=False, skip_norm=False,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.controlnet_pre = DiTControlNetEmbed(in_chans=cond_in, out_chans=embed_dim,
                                                 blocks=cond_blocks,
                                                 cond_mask=cond_mask, 
                                                 cond_mask_prob=cond_mask_prob,
                                                 cond_mask_ratio=cond_mask_ratio,
                                                 cond_mask_span=cond_mask_span)

        controlnet_zero_blocks = []
        for i in range(depth // 2):
            block = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(block.weight)
            nn.init.zeros_(block.bias)
            controlnet_zero_blocks.append(block)
        self.controlnet_zero_blocks = nn.ModuleList(controlnet_zero_blocks)

        print('ControlNet ready \n')

    def set_trainable(self):
        for param in self.parameters():
            param.requires_grad = False

        # only train input_proj, blocks, and output_proj
        for module_name in ['controlnet_pre', 'in_blocks', 'controlnet_zero_blocks']:
            module = getattr(self, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
                module.train()
            else:
                print(f'\n!!!warning missing trainable blocks: {module_name}!!!\n')

    def forward(self, x, timesteps, context,
                x_mask=None, context_mask=None,
                cls_token=None,
                condition=None, cond_mask_infer=None,
                conditioning_scale=1.0):
        # make it compatible with int time step during inference
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0]).to(x.device, dtype=torch.long)

        x = self.patch_embed(x)
        # add condition to x
        condition = self.controlnet_pre(condition)
        x = x + condition
        x = self.x_pe(x)

        B, L, D = x.shape

        if self.use_context:
            context_token = self.context_embed(context)
            context_token = self.context_pe(context_token)
            if self.context_fusion == 'concat' or self.context_fusion == 'joint':
                x, x_mask = self._concat_x_context(x=x, context=context_token,
                                                   x_mask=x_mask,
                                                   context_mask=context_mask)
                context_token, context_mask = None, None
        else:
            context_token, context_mask = None, None

        time_token = self.time_embed(timesteps)
        if self.cls_embed:
            cls_token = self.cls_embed(cls_token)
        time_ada = None
        if self.use_adanorm:
            if self.cls_embed:
                time_token = time_token + cls_token
            time_token = self.time_act(time_token)
            if self.time_ada is not None:
                time_ada = self.time_ada(time_token)
        else:
            time_token = time_token.unsqueeze(dim=1)
            if self.cls_embed:
                cls_token = cls_token.unsqueeze(dim=1)
                time_token = torch.cat([time_token, cls_token], dim=1)
            time_token = self.time_pe(time_token)
            x = torch.cat((time_token, x), dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [torch.ones(B, time_token.shape[1], device=x_mask.device).bool(),
                     x_mask], dim=1)
            time_token = None

        skips = []
        for blk in self.in_blocks:
            x = blk(x=x, time_token=time_token, time_ada=time_ada,
                    skip=None, context=context_token,
                    x_mask=x_mask, context_mask=context_mask,
                    extras=self.extras)
            skips.append(x)

        controlnet_skips = []
        for skip, controlnet_block in zip(skips, self.controlnet_zero_blocks):
            controlnet_skips.append(controlnet_block(skip) * conditioning_scale)

        return controlnet_skips