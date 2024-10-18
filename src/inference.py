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
def inference(autoencoder, unet, gt, gt_mask,
              tokenizer, text_encoder,
              params, noise_scheduler,
              text_raw, neg_text=None,
              audio_frames=500,
              guidance_scale=3, guidance_rescale=0.0,
              ddim_steps=50, eta=1, random_seed=2024,
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
            
            if gt is not None:
                gt_combined = torch.cat([gt, gt], dim=0)
                gt_mask_combined = torch.cat([gt_mask, gt_mask], dim=0)
            else:
                gt_combined = None
                gt_mask_combined = None
            
            output_combined, _ = unet(latents_combined, t, text_combined, context_mask=text_mask_combined, 
                                      cls_token=None, gt=gt_combined, mae_mask_infer=gt_mask_combined)
            output_text, output_uncond = torch.chunk(output_combined, 2, dim=0)

            output_pred = output_uncond + guidance_scale * (output_text - output_uncond)
            if guidance_rescale > 0.0:
                output_pred = rescale_noise_cfg(output_pred, output_text,
                                                guidance_rescale=guidance_rescale)
        else:
            output_pred, mae_mask = unet(latents, t, text, context_mask=text_mask,
                                         cls_token=None, gt=gt, mae_mask_infer=gt_mask)

        latents = noise_scheduler.step(model_output=output_pred, timestep=t, 
                                       sample=latents,
                                       eta=eta, generator=generator).prev_sample

    pred = scale_shift_re(latents, params['autoencoder']['scale'],
                          params['autoencoder']['shift'])
    if gt is not None:
        pred[~gt_mask] = gt[~gt_mask]
    pred_wav = autoencoder(embedding=pred)
    return pred_wav


@torch.no_grad()
def eval_udit(autoencoder, unet,
              tokenizer, text_encoder,
              params, noise_scheduler,
              val_df, subset,
              audio_frames, mae=False,
              guidance_scale=3, guidance_rescale=0.0,
              ddim_steps=50, eta=1, random_seed=2023,
              device='cuda',
              epoch=0, save_path='logs/eval/', val_num=5):
    val_df = pd.read_csv(val_df)
    val_df = val_df[val_df['split'] == subset]
    if mae:
        val_df = val_df[val_df['audio_length'] != 0]

    save_path = save_path + str(epoch) + '/'
    os.makedirs(save_path, exist_ok=True)

    for i in tqdm(range(len(val_df))):
        row = val_df.iloc[i]
        text = [row['caption']]
        if mae:
            audio_path = params['data']['val_dir'] + str(row['audio_path'])
            gt, sr = librosa.load(audio_path, sr=params['data']['sr'])
            gt = gt / (np.max(np.abs(gt)) + 1e-9)
            sf.write(save_path + text[0] + '_gt.wav', gt, samplerate=params['data']['sr'])
            num_samples = 10 * sr
            if len(gt) < num_samples:
                padding = num_samples - len(gt)
                gt = np.pad(gt, (0, padding), 'constant')
            else:
                gt = gt[:num_samples]
            gt = torch.tensor(gt).unsqueeze(0).unsqueeze(1).to(device)
            gt = autoencoder(audio=gt)
            B, D, L = gt.shape
            mask_len = int(L * 0.2)
            gt_mask = torch.zeros(B, D, L).to(device)
            for _ in range(2):
                start = random.randint(0, L - mask_len)
                gt_mask[:, :, start:start + mask_len] = 1
            gt_mask = gt_mask.bool()
        else:
            gt = None
            gt_mask = None

        pred = inference(autoencoder, unet, gt, gt_mask,
                         tokenizer, text_encoder, 
                         params, noise_scheduler,
                         text, neg_text=None,
                         audio_frames=audio_frames,
                         guidance_scale=guidance_scale, guidance_rescale=guidance_rescale,
                         ddim_steps=ddim_steps, eta=eta, random_seed=random_seed,
                         device=device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

        sf.write(save_path + text[0] + '.wav', pred, samplerate=params['data']['sr'])

        if i + 1 >= val_num:
            break
