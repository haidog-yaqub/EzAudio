import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .utils.attention import Attention, JointAttention
from .utils.modules import unpatchify, FeedForward
from .utils.modules import film_modulate


class AdaLN(nn.Module):
    def __init__(self, dim, ada_mode='ada', r=None, alpha=None):
        super().__init__()
        self.ada_mode = ada_mode
        self.scale_shift_table = None
        if ada_mode == 'ada':
            # move nn.silu outside
            self.time_ada = nn.Linear(dim, 6 * dim, bias=True)
        elif ada_mode == 'ada_single':
            # adaln used in pixel-art alpha
            self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
        elif ada_mode in ['ada_solo', 'ada_sola_bias']:
            self.lora_a = nn.Linear(dim, r * 6, bias=False)
            self.lora_b = nn.Linear(r * 6, dim * 6, bias=False)
            self.scaling = alpha / r
            if ada_mode == 'ada_sola_bias':
                # take bias out for consistency
                self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
        else:
            raise NotImplementedError

    def forward(self, time_token=None, time_ada=None):
        if self.ada_mode == 'ada':
            assert time_ada is None
            B = time_token.shape[0]
            time_ada = self.time_ada(time_token).reshape(B, 6, -1)
        elif self.ada_mode == 'ada_single':
            B = time_ada.shape[0]
            time_ada = time_ada.reshape(B, 6, -1)
            time_ada = self.scale_shift_table[None] + time_ada
        elif self.ada_mode in ['ada_sola', 'ada_sola_bias']:
            B = time_ada.shape[0]
            time_ada_lora = self.lora_b(self.lora_a(time_token)) * self.scaling
            time_ada = time_ada + time_ada_lora
            time_ada = time_ada.reshape(B, 6, -1)
            if self.scale_shift_table is not None:
                time_ada = self.scale_shift_table[None] + time_ada
        else:
            raise NotImplementedError
        return time_ada


class DiTBlock(nn.Module):
    """
    A modified PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, dim, context_dim=None,
                 num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm=None,
                 act_layer='gelu', norm_layer=nn.LayerNorm,
                 time_fusion='none',
                 ada_sola_rank=None, ada_sola_alpha=None,
                 skip=False, skip_norm=False,
                 rope_mode='none',
                 context_norm=False,
                 use_checkpoint=False):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              qk_norm=qk_norm,
                              rope_mode=rope_mode)

        if context_dim is not None:
            self.use_context = True
            self.cross_attn = Attention(dim=dim,
                                        num_heads=num_heads,
                                        context_dim=context_dim,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        qk_norm=qk_norm,
                                        rope_mode='none')
            self.norm2 = norm_layer(dim)
            if context_norm:
                self.norm_context = norm_layer(context_dim)
            else:
                self.norm_context = nn.Identity()
        else:
            self.use_context = False

        self.norm3 = norm_layer(dim)
        self.mlp = FeedForward(dim=dim, mult=mlp_ratio,
                               activation_fn=act_layer, dropout=0)

        self.use_adanorm = True if time_fusion != 'token' else False
        if self.use_adanorm:
            self.adaln = AdaLN(dim, ada_mode=time_fusion,
                               r=ada_sola_rank, alpha=ada_sola_alpha)
        if skip:
            self.skip_norm = norm_layer(2 * dim) if skip_norm else nn.Identity()
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        self.use_checkpoint = use_checkpoint

    def forward(self, x, time_token=None, time_ada=None,
                skip=None, context=None,
                x_mask=None, context_mask=None, extras=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,
                              time_token, time_ada, skip, context,
                              x_mask, context_mask, extras,
                              use_reentrant=False)
        else:
            return self._forward(x,
                                 time_token, time_ada, skip, context,
                                 x_mask, context_mask, extras)

    def _forward(self, x, time_token=None, time_ada=None,
                 skip=None, context=None,
                 x_mask=None, context_mask=None, extras=None):
        B, T, C = x.shape
        if self.skip_linear is not None:
            assert skip is not None
            cat = torch.cat([x, skip], dim=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        if self.use_adanorm:
            time_ada = self.adaln(time_token, time_ada)
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = time_ada.chunk(6, dim=1)

        # self attention
        if self.use_adanorm:
            x_norm = film_modulate(self.norm1(x), shift=shift_msa,
                                   scale=scale_msa)
            x = x + (1 - gate_msa) * self.attn(x_norm, context=None,
                                               context_mask=x_mask,
                                               extras=extras)
        else:
            x = x + self.attn(self.norm1(x), context=None, context_mask=x_mask,
                              extras=extras)

        # cross attention
        if self.use_context:
            assert context is not None
            x = x + self.cross_attn(x=self.norm2(x),
                                    context=self.norm_context(context),
                                    context_mask=context_mask, extras=extras)

        # mlp
        if self.use_adanorm:
            x_norm = film_modulate(self.norm3(x), shift=shift_mlp, scale=scale_mlp)
            x = x + (1 - gate_mlp) * self.mlp(x_norm)
        else:
            x = x + self.mlp(self.norm3(x))

        return x



class FinalBlock(nn.Module):
    def __init__(self, embed_dim, patch_size, in_chans,
                 img_size,
                 input_type='2d',
                 norm_layer=nn.LayerNorm,
                 use_conv=True,
                 use_adanorm=True):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.input_type = input_type

        self.norm = norm_layer(embed_dim)
        if use_adanorm:
            self.use_adanorm = True
        else:
            self.use_adanorm = False

        if input_type == '2d':
            self.patch_dim = patch_size ** 2 * in_chans
            self.linear = nn.Linear(embed_dim, self.patch_dim, bias=True)
            if use_conv:
                self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 
                                             3, padding=1)
            else:
                self.final_layer = nn.Identity()

        elif input_type == '1d':
            self.patch_dim = patch_size * in_chans
            self.linear = nn.Linear(embed_dim, self.patch_dim, bias=True)
            if use_conv:
                self.final_layer = nn.Conv1d(self.in_chans, self.in_chans, 
                                             3, padding=1)
            else:
                self.final_layer = nn.Identity()

    def forward(self, x, time_ada=None, extras=0):
        B, T, C = x.shape
        x = x[:, extras:, :]
        # only handle generation target
        if self.use_adanorm:
            shift, scale = time_ada.reshape(B, 2, -1).chunk(2, dim=1)
            x = film_modulate(self.norm(x), shift, scale)
        else:
            x = self.norm(x)
        x = self.linear(x)
        x = unpatchify(x, self.in_chans, self.input_type, self.img_size)
        x = self.final_layer(x)
        return x