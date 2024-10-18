import torch
import torch.nn as nn
import torch.utils.checkpoint
import einops
from einops import rearrange, repeat
from inspect import isfunction
from .rotary import RotaryEmbedding

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


def create_mask(q, k, q_mask=None, k_mask=None):
    def default(val, d):
        return val if val is not None else (d() if isfunction(d) else d)

    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    q_mask = default(q_mask, torch.ones((b, i), device=device, dtype=torch.bool))
    k_mask = default(k_mask, torch.ones((b, j), device=device, dtype=torch.bool))
    attn_mask = rearrange(q_mask, 'b i -> b 1 i 1') * rearrange(k_mask, 'b j -> b 1 1 j')
    return attn_mask


class Attention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        context_dim = dim if context_dim is None else context_dim

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rope = use_rope
        if self.use_rope:
            self.rotary = RotaryEmbedding(dim=head_dim)

    def forward(self, x, context=None, context_mask=None):
        B, L, C = x.shape
        q = self.to_q(x)
        if context is None:
            context = x
        else:
            assert self.use_rope is False

        k = self.to_k(context)
        v = self.to_v(context)

        if context_mask is not None:
            mask_binary = create_mask(x, context, None, context_mask)
        else:
            mask_binary = None

        q = einops.rearrange(q, 'B L (H D) -> B H L D', H=self.num_heads).float()
        k = einops.rearrange(k, 'B L (H D) -> B H L D', H=self.num_heads).float()
        v = einops.rearrange(v, 'B L (H D) -> B H L D', H=self.num_heads).float()

        if self.use_rope:
            q, k = self.rotary(q=q, k=k)

        if ATTENTION_MODE == 'flash':
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                 dropout_p=self.attn_drop_p,
                                                                 attn_mask=mask_binary)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = add_mask(attn, mask_binary) if mask_binary is not None else attn
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplementedError

        x = self.proj(x)
        x = self.proj_drop(x)
        return x