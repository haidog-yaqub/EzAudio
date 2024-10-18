import os
import random
import pandas as pd
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from .utils import scale_shift_re


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@torch.no_grad()
def inference(autoencoder, unet, controlnet,
              gt, gt_mask, condition,
              tokenizer, text_encoder,
              params, noise_scheduler,
              text_raw, neg_text=None,
              audio_frames=500,
              guidance_scale=3, guidance_rescale=0.0,
              ddim_steps=50, eta=1, random_seed=2024,
              conditioning_scale=1.0,
              device='cuda',
              ):
    if neg_text is None:
        neg_text = [""]
    if tokenizer is not None:
        text_batch = tokenizer(text_raw,
                               max_length=params['text_encoder']['max_length'],
                               padding="max_length", truncation=True, return_tensors="pt")
        text, text_mask = text_batch.input_ids.to(device), text_batch.attention_mask.to(device).bool()
        text = text_encoder(input_ids=text, attention_mask=text_mask).last_hidden_state

        uncond_text_batch = tokenizer(neg_text,
                                      max_length=params['text_encoder']['max_length'],
                                      padding="max_length", truncation=True, return_tensors="pt")
        uncond_text, uncond_text_mask = uncond_text_batch.input_ids.to(device), uncond_text_batch.attention_mask.to(device).bool()
        uncond_text = text_encoder(input_ids=uncond_text,
                                   attention_mask=uncond_text_mask).last_hidden_state
    else:
        text, text_mask = None, None
        guidance_scale = None

    codec_dim = params['model']['out_chans']
    unet.eval()
    controlnet.eval()

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = torch.Generator(device=device)
        generator.seed()

    noise_scheduler.set_timesteps(ddim_steps)

    # init noise
    noise = torch.randn((1, codec_dim, audio_frames), generator=generator, device=device)
    latents = noise

    for t in noise_scheduler.timesteps:
        latents = noise_scheduler.scale_model_input(latents, t)

        if guidance_scale:
            latents_combined = torch.cat([latents, latents], dim=0)
            text_combined = torch.cat([text, uncond_text], dim=0)
            text_mask_combined = torch.cat([text_mask, uncond_text_mask], dim=0)
            condition_combined = torch.cat([condition, condition], dim=0)

            if gt is not None:
                gt_combined = torch.cat([gt, gt], dim=0)
                gt_mask_combined = torch.cat([gt_mask, gt_mask], dim=0)
            else:
                gt_combined = None
                gt_mask_combined = None

            x, _ = unet(latents_combined, t, text_combined, context_mask=text_mask_combined,
                        cls_token=None, gt=gt_combined, mae_mask_infer=gt_mask_combined, 
                        forward_model=False)
            controlnet_skips = controlnet(x, t, text_combined,
                                          context_mask=text_mask_combined,
                                          cls_token=None,
                                          condition=condition_combined,
                                          conditioning_scale=conditioning_scale)
            output_combined = unet.model(x, t, text_combined,
                                         context_mask=text_mask_combined,
                                         cls_token=None, controlnet_skips=controlnet_skips)

            output_text, output_uncond = torch.chunk(output_combined, 2, dim=0)

            output_pred = output_uncond + guidance_scale * (output_text - output_uncond)
            if guidance_rescale > 0.0:
                output_pred = rescale_noise_cfg(output_pred, output_text,
                                                guidance_rescale=guidance_rescale)
        else:
            x, _ = unet(latents, t, text, context_mask=text_mask,
                        cls_token=None, gt=gt, mae_mask_infer=gt_mask,
                        forward_model=False)
            controlnet_skips = controlnet(x, t, text,
                                          context_mask=text_mask,
                                          cls_token=None,
                                          condition=condition,
                                          conditioning_scale=conditioning_scale)
            output_pred = unet.model(x, t, text,
                                     context_mask=text_mask,
                                     cls_token=None, controlnet_skips=controlnet_skips)

        latents = noise_scheduler.step(model_output=output_pred, timestep=t,
                                       sample=latents,
                                       eta=eta, generator=generator).prev_sample

    pred = scale_shift_re(latents, params['autoencoder']['scale'],
                          params['autoencoder']['shift'])
    if gt is not None:
        pred[~gt_mask] = gt[~gt_mask]
    pred_wav = autoencoder(embedding=pred)
    return pred_wav