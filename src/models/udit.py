import torch
import torch.nn as nn
import torch.utils.checkpoint
import math
from .utils.modules import PatchEmbed, TimestepEmbedder
from .utils.modules import PE_wrapper, RMSNorm
from .blocks import DiTBlock, FinalBlock


class UDiT(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,
                 input_type='2d', out_chans=None,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm=None,
                 act_layer='gelu', norm_layer='layernorm',
                 context_norm=False,
                 use_checkpoint=False,
                 # time fusion ada or token
                 time_fusion='token',
                 ada_sola_rank=None, ada_sola_alpha=None,
                 cls_dim=None,
                 # max length is only used for concat
                 context_dim=768, context_fusion='concat',
                 context_max_length=128, context_pe_method='sinu',
                 pe_method='abs', rope_mode='none',
                 use_conv=True,
                 skip=True, skip_norm=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

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
        elif time_fusion in ['ada', 'ada_single', 'ada_sola', 'ada_sola_bias']:
            self.use_adanorm = True
            # aviod  repetitive silu for each adaln block
            self.time_act = nn.SiLU()
            self.extras = 0
            self.time_ada_final = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
            if time_fusion in ['ada_single', 'ada_sola', 'ada_sola_bias']:
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


        Block = DiTBlock
        self.use_skip = skip

        # norm layers
        if norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm_layer == 'rmsnorm':
            norm_layer = RMSNorm
        else:
            raise NotImplementedError

        print(f'use long skip connection: {skip}')
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
                act_layer=act_layer, norm_layer=norm_layer,
                time_fusion=time_fusion,
                ada_sola_rank=ada_sola_rank, ada_sola_alpha=ada_sola_alpha,
                skip=False, skip_norm=False,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
            dim=embed_dim, context_dim=context_dim, num_heads=num_heads, 
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
            act_layer=act_layer, norm_layer=norm_layer,
            time_fusion=time_fusion,
            ada_sola_rank=ada_sola_rank, ada_sola_alpha=ada_sola_alpha,
            skip=False, skip_norm=False,
            rope_mode=self.rope,
            context_norm=context_norm,
            use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
                act_layer=act_layer, norm_layer=norm_layer,
                time_fusion=time_fusion,
                ada_sola_rank=ada_sola_rank, ada_sola_alpha=ada_sola_alpha,
                skip=skip, skip_norm=skip_norm,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        # FinalLayer block
        self.use_conv = use_conv
        self.final_block = FinalBlock(embed_dim=embed_dim,
                                      patch_size=patch_size,
                                      img_size=img_size,
                                      in_chans=out_chans,
                                      input_type=input_type,
                                      norm_layer=norm_layer,
                                      use_conv=use_conv,
                                      use_adanorm=self.use_adanorm)
        self.initialize_weights()

    def _init_ada(self):
        if self.time_fusion == 'ada':
            nn.init.constant_(self.time_ada_final.weight, 0)
            nn.init.constant_(self.time_ada_final.bias, 0)
            for block in self.in_blocks:
                nn.init.constant_(block.adaln.time_ada.weight, 0)
                nn.init.constant_(block.adaln.time_ada.bias, 0)
            nn.init.constant_(self.mid_block.adaln.time_ada.weight, 0)
            nn.init.constant_(self.mid_block.adaln.time_ada.bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.adaln.time_ada.weight, 0)
                nn.init.constant_(block.adaln.time_ada.bias, 0)
        elif self.time_fusion == 'ada_single':
            nn.init.constant_(self.time_ada.weight, 0)
            nn.init.constant_(self.time_ada.bias, 0)
            nn.init.constant_(self.time_ada_final.weight, 0)
            nn.init.constant_(self.time_ada_final.bias, 0)
        elif self.time_fusion in ['ada_sola', 'ada_sola_bias']:
            nn.init.constant_(self.time_ada.weight, 0)
            nn.init.constant_(self.time_ada.bias, 0)
            nn.init.constant_(self.time_ada_final.weight, 0)
            nn.init.constant_(self.time_ada_final.bias, 0)
            for block in self.in_blocks:
                nn.init.kaiming_uniform_(block.adaln.lora_a.weight,
                                         a=math.sqrt(5))
                nn.init.constant_(block.adaln.lora_b.weight, 0)
            nn.init.kaiming_uniform_(self.mid_block.adaln.lora_a.weight,
                                     a=math.sqrt(5))
            nn.init.constant_(self.mid_block.adaln.lora_b.weight, 0)
            for block in self.out_blocks:
                nn.init.kaiming_uniform_(block.adaln.lora_a.weight,
                                         a=math.sqrt(5))
                nn.init.constant_(block.adaln.lora_b.weight, 0)

    def initialize_weights(self):
        # Basic init for all layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # init patch Conv like Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Zero-out AdaLN
        if self.use_adanorm:
            self._init_ada()

        # Zero-out Cross Attention
        if self.context_cross:
            for block in self.in_blocks:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)
            nn.init.constant_(self.mid_block.cross_attn.proj.weight, 0)
            nn.init.constant_(self.mid_block.cross_attn.proj.bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out cls embedding
        if self.cls_embed:
            if self.use_adanorm:
                nn.init.constant_(self.cls_embed[-1].weight, 0)
                nn.init.constant_(self.cls_embed[-1].bias, 0)

        # Zero-out Output
        # might not zero-out this when using v-prediction
        # it could be good when using noise-prediction
        # nn.init.constant_(self.final_block.linear.weight, 0)
        # nn.init.constant_(self.final_block.linear.bias, 0)
        # if self.use_conv:
        #     nn.init.constant_(self.final_block.final_layer.weight.data, 0)
        #     nn.init.constant_(self.final_block.final_layer.bias, 0)

        # init out Conv
        if self.use_conv:
            nn.init.xavier_uniform_(self.final_block.final_layer.weight)
            nn.init.constant_(self.final_block.final_layer.bias, 0)

    def _concat_x_context(self, x, context, x_mask=None, context_mask=None):
        assert context.shape[-2] == self.context_max_length
        # Check if either x_mask or context_mask is provided
        B = x.shape[0]
        # Create default masks if they are not provided
        if x_mask is None:
            x_mask = torch.ones(B, x.shape[-2], device=x.device).bool()
        if context_mask is None:
            context_mask = torch.ones(B, context.shape[-2],
                                      device=context.device).bool()
        # Concatenate the masks along the second dimension (dim=1)
        x_mask = torch.cat([context_mask, x_mask], dim=1)
        # Concatenate context and x along the second dimension (dim=1)
        x = torch.cat((context, x), dim=1)
        return x, x_mask

    def forward(self, x, timesteps, context,
                x_mask=None, context_mask=None,
                cls_token=None, controlnet_skips=None,
               ):
        # make it compatible with int time step during inference
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0]).to(x.device, dtype=torch.long)

        x = self.patch_embed(x)
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
        time_ada_final = None
        if self.use_adanorm:
            if self.cls_embed:
                time_token = time_token + cls_token
            time_token = self.time_act(time_token)
            time_ada_final = self.time_ada_final(time_token)
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
            if self.use_skip:
                skips.append(x)

        x = self.mid_block(x=x, time_token=time_token, time_ada=time_ada,
                           skip=None, context=context_token,
                           x_mask=x_mask, context_mask=context_mask,
                           extras=self.extras)
        for blk in self.out_blocks:
            if self.use_skip:
                skip = skips.pop()
                if controlnet_skips:
                    # add to skip like u-net controlnet
                    skip = skip + controlnet_skips.pop()
            else:
                skip = None
                if controlnet_skips:
                    # directly add to x
                    x = x + controlnet_skips.pop()

            x = blk(x=x, time_token=time_token, time_ada=time_ada,
                    skip=skip, context=context_token,
                    x_mask=x_mask, context_mask=context_mask,
                    extras=self.extras)

        x = self.final_block(x, time_ada=time_ada_final, extras=self.extras)

        return x