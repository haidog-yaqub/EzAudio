import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import einops
from einops import rearrange, repeat
from inspect import isfunction
from .rotary import RotaryEmbedding
from .modules import RMSNorm


if hasattr(nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def add_mask(sim, mask):
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def create_mask(q_shape, k_shape, device, q_mask=None, k_mask=None):
    def default(val, d):
        return val if val is not None else (d() if isfunction(d) else d)
    b, i, j, device = q_shape[0], q_shape[-2], k_shape[-2], device
    q_mask = default(q_mask, torch.ones((b, i), device=device, dtype=torch.bool))
    k_mask = default(k_mask, torch.ones((b, j), device=device, dtype=torch.bool))
    attn_mask = rearrange(q_mask, 'b i -> b 1 i 1') * rearrange(k_mask, 'b j -> b 1 1 j')
    return attn_mask


class Attention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8,
                 qkv_bias=False, qk_scale=None, qk_norm=None,
                 attn_drop=0., proj_drop=0., rope_mode='none'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if context_dim is None:
            self.cross_attn = False
        else:
            self.cross_attn = True

        context_dim = dim if context_dim is None else context_dim

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, dim, bias=qkv_bias)

        if qk_norm is None:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        elif qk_norm == 'layernorm':
            self.norm_q = nn.LayerNorm(head_dim)
            self.norm_k = nn.LayerNorm(head_dim)
        elif qk_norm == 'rmsnorm':
            self.norm_q = RMSNorm(head_dim)
            self.norm_k = RMSNorm(head_dim)
        else:
            raise NotImplementedError

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.cross_attn:
            assert rope_mode == 'none'
        self.rope_mode = rope_mode
        if self.rope_mode == 'shared' or self.rope_mode == 'x_only':
            self.rotary = RotaryEmbedding(dim=head_dim)
        elif self.rope_mode == 'dual':
            self.rotary_x = RotaryEmbedding(dim=head_dim)
            self.rotary_c = RotaryEmbedding(dim=head_dim)

    def _rotary(self, q, k, extras):
        if self.rope_mode == 'shared':
            q, k = self.rotary(q=q, k=k)
        elif self.rope_mode == 'x_only':
            q_x, k_x = self.rotary(q=q[:, :, extras:, :], k=k[:, :, extras:, :])
            q_c, k_c = q[:, :, :extras, :], k[:, :, :extras, :]
            q = torch.cat((q_c, q_x), dim=2)
            k = torch.cat((k_c, k_x), dim=2)
        elif self.rope_mode == 'dual':
            q_x, k_x = self.rotary_x(q=q[:, :, extras:, :], k=k[:, :, extras:, :])
            q_c, k_c = self.rotary_c(q=q[:, :, :extras, :], k=k[:, :, :extras, :])
            q = torch.cat((q_c, q_x), dim=2)
            k = torch.cat((k_c, k_x), dim=2)
        elif self.rope_mode == 'none':
            pass
        else:
            raise NotImplementedError
        return q, k

    def _attn(self, q, k, v, mask_binary):
        if ATTENTION_MODE == 'flash':
            x = F.scaled_dot_product_attention(q, k, v,
                                               dropout_p=self.attn_drop_p,
                                               attn_mask=mask_binary)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = add_mask(attn, mask_binary) if mask_binary is not None else attn
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        else:
            raise NotImplementedError
        return x

    def forward(self, x, context=None, context_mask=None, extras=0):
        B, L, C = x.shape
        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        if context_mask is not None:
            mask_binary = create_mask(x.shape, context.shape,
                                      x.device, None, context_mask)
        else:
            mask_binary = None

        q = einops.rearrange(q, 'B L (H D) -> B H L D', H=self.num_heads)
        k = einops.rearrange(k, 'B L (H D) -> B H L D', H=self.num_heads)
        v = einops.rearrange(v, 'B L (H D) -> B H L D', H=self.num_heads)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q, k = self._rotary(q, k, extras)

        x = self._attn(q, k, v, mask_binary)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class JointAttention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 qkv_bias=False, qk_scale=None, qk_norm=None,
                 attn_drop=0., proj_drop=0.,
                 rope_mode='none'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_qx, self.to_kx, self.to_vx = self._make_qkv_layers(dim, qkv_bias)
        self.to_qc, self.to_kc, self.to_vc = self._make_qkv_layers(dim, qkv_bias)

        self.norm_qx, self.norm_kx = self._make_norm_layers(qk_norm, head_dim)
        self.norm_qc, self.norm_kc = self._make_norm_layers(qk_norm, head_dim)

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_drop_x = nn.Dropout(proj_drop)

        self.proj_c = nn.Linear(dim, dim)
        self.proj_drop_c = nn.Dropout(proj_drop)

        self.rope_mode = rope_mode
        if self.rope_mode == 'shared' or self.rope_mode == 'x_only':
            self.rotary = RotaryEmbedding(dim=head_dim)
        elif self.rope_mode == 'dual':
            self.rotary_x = RotaryEmbedding(dim=head_dim)
            self.rotary_c = RotaryEmbedding(dim=head_dim)

    def _make_qkv_layers(self, dim, qkv_bias):
        return (nn.Linear(dim, dim, bias=qkv_bias),
                nn.Linear(dim, dim, bias=qkv_bias),
                nn.Linear(dim, dim, bias=qkv_bias))

    def _make_norm_layers(self, qk_norm, head_dim):
        if qk_norm is None:
            norm_q = nn.Identity()
            norm_k = nn.Identity()
        elif qk_norm == 'layernorm':
            norm_q = nn.LayerNorm(head_dim)
            norm_k = nn.LayerNorm(head_dim)
        elif qk_norm == 'rmsnorm':
            norm_q = RMSNorm(head_dim)
            norm_k = RMSNorm(head_dim)
        else:
            raise NotImplementedError
        return norm_q, norm_k

    def _rotary(self, q, k, extras):
        if self.rope_mode == 'shared':
            q, k = self.rotary(q=q, k=k)
        elif self.rope_mode == 'x_only':
            q_x, k_x = self.rotary(q=q[:, :, extras:, :], k=k[:, :, extras:, :])
            q_c, k_c = q[:, :, :extras, :], k[:, :, :extras, :]
            q = torch.cat((q_c, q_x), dim=2)
            k = torch.cat((k_c, k_x), dim=2)
        elif self.rope_mode == 'dual':
            q_x, k_x = self.rotary_x(q=q[:, :, extras:, :], k=k[:, :, extras:, :])
            q_c, k_c = self.rotary_c(q=q[:, :, :extras, :], k=k[:, :, :extras, :])
            q = torch.cat((q_c, q_x), dim=2)
            k = torch.cat((k_c, k_x), dim=2)
        elif self.rope_mode == 'none':
            pass
        else:
            raise NotImplementedError
        return q, k

    def _attn(self, q, k, v, mask_binary):
        if ATTENTION_MODE == 'flash':
            x = F.scaled_dot_product_attention(q, k, v,
                                               dropout_p=self.attn_drop_p,
                                               attn_mask=mask_binary)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = add_mask(attn, mask_binary) if mask_binary is not None else attn
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        else:
            raise NotImplementedError
        return x

    def _cat_mask(self, x, context, x_mask=None, context_mask=None):
        B = x.shape[0]
        if x_mask is None:
            x_mask = torch.ones(B, x.shape[-2], device=x.device).bool()
        if context_mask is None:
            context_mask = torch.ones(B, context.shape[-2], device=context.device).bool()
        mask = torch.cat([context_mask, x_mask], dim=1)
        return mask

    def forward(self, x, context, x_mask=None, context_mask=None, extras=0):
        B, Lx, C = x.shape
        _, Lc, _ = context.shape
        if x_mask is not None or context_mask is not None:
            mask = self._cat_mask(x, context,
                                  x_mask=x_mask,
                                  context_mask=context_mask)
            shape = [B, Lx+Lc, C]
            mask_binary = create_mask(q_shape=shape, k_shape=shape,
                                      device=x.device,
                                      q_mask=None, k_mask=mask)
        else:
            mask_binary = None

        qx, kx, vx = self.to_qx(x), self.to_kx(x), self.to_vx(x)
        qc, kc, vc = self.to_qc(context), self.to_kc(context), self.to_vc(context)

        qx, kx, vx = map(lambda t: einops.rearrange(t, 'B L (H D) -> B H L D',
                                                    H=self.num_heads), [qx, kx, vx])
        qc, kc, vc = map(lambda t: einops.rearrange(t, 'B L (H D) -> B H L D',
                                                    H=self.num_heads), [qc, kc, vc])

        qx, kx = self.norm_qx(qx), self.norm_kx(kx)
        qc, kc = self.norm_qc(qc), self.norm_kc(kc)

        q, k, v = (torch.cat([qc, qx], dim=2),
                   torch.cat([kc, kx], dim=2),
                   torch.cat([vc, vx], dim=2))

        q, k = self._rotary(q, k, extras)

        x = self._attn(q, k, v, mask_binary)

        context, x = x[:, :Lc, :], x[:, Lc:, :]

        x = self.proj_x(x)
        x = self.proj_drop_x(x)

        context = self.proj_c(context)
        context = self.proj_drop_c(context)

        return x, context