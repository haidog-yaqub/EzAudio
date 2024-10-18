import torch
import numpy as np
import yaml
import os


def load_yaml_with_includes(yaml_file):
    def loader_with_include(loader, node):
        # Load the included file
        include_path = os.path.join(os.path.dirname(yaml_file), loader.construct_scalar(node))
        with open(include_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    yaml.add_constructor('!include', loader_with_include, Loader=yaml.FullLoader)

    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def scale_shift(x, scale, shift):
    return (x+shift) * scale


def scale_shift_re(x, scale, shift):
    return (x/scale) - shift


def align_seq(source, target_length, mapping_method='hard'):
    source_len = source.shape[1]
    if mapping_method == 'hard':
        mapping_idx = np.round(np.arange(target_length) * source_len / target_length)
        output = source[:, mapping_idx]
    else:
        # TBD
        raise NotImplementedError

    return output


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR

    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion
    Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion
    # Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


if __name__ == "__main__":

    a = torch.rand(2, 10)
    target_len = 15

    b = align_seq(a, target_len)