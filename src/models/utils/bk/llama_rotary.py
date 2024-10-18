import torch
from typing import Tuple
from rotary import RotaryEmbedding
import time


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor,
                          x: torch.Tensor,):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def compute_rope(q, freqs_cis):
    return q * freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq1, xq2 = xq.chunk(2, dim=-1)
    xq_ = torch.view_as_complex(torch.stack((xq1, xq2), dim=-1).float())

    xk1, xk2 = xk.chunk(2, dim=-1)
    xk_ = torch.view_as_complex(torch.stack((xk1, xk2), dim=-1).float())

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(compute_rope(xq_, freqs_cis)).flatten(3)
    xk_out = torch.view_as_real(compute_rope(xk_, freqs_cis)).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == '__main__':
    # Move data to CUDA
    freq_cis = precompute_freqs_cis(4, 5).cuda()
    x = torch.rand(1, 5, 1, 4).cuda()
    y = torch.rand(1, 5, 1, 4).cuda()

    # First method
    start_time = time.time()
    for _ in range(20000):
        x1, y1 = apply_rotary_emb(x, y, freq_cis)
    end_time = time.time()
    print(f"Method 1 time cost: {end_time - start_time} seconds")

    # Prepare data for the second method
    x = x.permute(0, 2, 1, 3)
    y = y.permute(0, 2, 1, 3)
    rope = RotaryEmbedding(4).cuda()

    # Second method
    start_time = time.time()
    for _ in range(20000):
        x2, y2 = rope(x, y)
    end_time = time.time()
    print(f"Method 2 time cost: {end_time - start_time} seconds")

    # Print the results
    print(x1)
    print(x2)