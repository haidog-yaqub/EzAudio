import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast
import math
import einops
from einops import rearrange, repeat
from inspect import isfunction
from .timm import trunc_normal_


# disable in checkpoint mode
# @torch.jit.script
def film_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, 
                 out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(
            self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def patchify(imgs, patch_size, input_type='2d'):
    if input_type == '2d':
        x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    elif input_type == '1d':
        x = einops.rearrange(imgs, 'B C (h p1) -> B h (p1 C)', p1=patch_size)
    return x


def unpatchify(x, channels=3, input_type='2d', img_size=None):
    if input_type == '2d':
        patch_size = int((x.shape[2] // channels) ** 0.5)
        # h = w = int(x.shape[1] ** .5)
        h, w = img_size[0] // patch_size, img_size[1] // patch_size
        assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
        x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h,
                             p1=patch_size, p2=patch_size)
    elif input_type == '1d':
        patch_size = int((x.shape[2] // channels))
        h = x.shape[1]
        assert patch_size * channels == x.shape[2]
        x = einops.rearrange(x, 'B h (p1 C) -> B C (h p1)', h=h, p1=patch_size)
    return x


class PatchEmbed(nn.Module):
    """
     Image to Patch Embedding
    """

    def __init__(self, patch_size, in_chans=3, embed_dim=768, input_type='2d'):
        super().__init__()
        self.patch_size = patch_size
        self.input_type = input_type
        if input_type == '2d':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        elif input_type == '1d':
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        if self.input_type == '2d':
            B, C, H, W = x.shape
            assert H % self.patch_size == 0 and W % self.patch_size == 0
        elif self.input_type == '1d':
            B, C, H = x.shape
            assert H % self.patch_size == 0

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalConvEmbedding(nn.Module):
    """
    Relative positional embedding used in HuBERT
    """

    def __init__(self, dim=768, kernel_size=128, groups=16):
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=True
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x):
        # B C T
        x = self.conv(x)
        x = F.gelu(x[:, :, :-1])
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, length):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.length = length
        self.dim = dim
        self.register_buffer('pe', self._generate_positional_encoding(length, dim))

    def _generate_positional_encoding(self, length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class PE_wrapper(nn.Module):
    def __init__(self, dim=768, method='abs', length=None, **kwargs):
        super().__init__()
        self.method = method
        if method == 'abs':
            # init absolute pe like UViT
            self.length = length
            self.abs_pe = nn.Parameter(torch.zeros(1, length, dim))
            trunc_normal_(self.abs_pe, std=.02)
        elif method == 'conv':
            self.conv_pe = PositionalConvEmbedding(dim=dim, **kwargs)
        elif method == 'sinu':
            self.sinu_pe = SinusoidalPositionalEncoding(dim=dim, length=length)
        elif method == 'none':
            # skip pe
            self.id = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.method == 'abs':
            _, L, _ = x.shape
            assert L <= self.length
            x = x + self.abs_pe[:, :L, :]
        elif self.method == 'conv':
            x = x + self.conv_pe(x)
        elif self.method == 'sinu':
            x = self.sinu_pe(x)
        elif self.method == 'none':
            x = self.id(x)
        else:
            raise NotImplementedError
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GELU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", 
                 bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32),
                      approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


# disable in checkpoint mode
# @torch.jit.script
def snake_beta(x, alpha, beta):
    return x + beta * torch.sin(x * alpha).pow(2)


class Snake(nn.Module):
    def __init__(self, dim_in, dim_out, bias,
                 alpha_trainable=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1, 1, dim_out))
        self.beta = nn.Parameter(torch.ones(1, 1, dim_out))
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        x = self.proj(x)
        x = snake_beta(x, self.alpha, self.beta)
        return x


class GESnake(nn.Module):
    def __init__(self, dim_in, dim_out, bias,
                 alpha_trainable=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.alpha = nn.Parameter(torch.ones(1, 1, dim_out))
        self.beta = nn.Parameter(torch.ones(1, 1, dim_out))
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * snake_beta(gate, self.alpha, self.beta)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        dropout=0.0,
        activation_fn="geglu",
        final_dropout=False,
        inner_dim=None,
        bias=True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "snake":
            act_fn = Snake(dim, inner_dim, bias=bias)
        elif activation_fn == "gesnake":
            act_fn = GESnake(dim, inner_dim, bias=bias)
        else:
            raise NotImplementedError

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states