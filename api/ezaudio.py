import os
import torch
import random
import numpy as np
import librosa
import soundfile as sf
from accelerate import Accelerator
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler
from src.models.conditioners import MaskDiT
from src.modules.autoencoder_wrapper import Autoencoder
from src.inference import inference
from src.utils import load_yaml_with_includes


# Load model and configs
def load_models(config_name, ckpt_path, vae_path, device):
    params = load_yaml_with_includes(config_name)

    # Load codec model
    autoencoder = Autoencoder(ckpt_path=vae_path,
                              model_type=params['autoencoder']['name'],
                              quantization_first=params['autoencoder']['q_first']).to(device)
    autoencoder.eval()

    # Load text encoder
    tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'])
    text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model']).to(device)
    text_encoder.eval()

    # Load main U-Net model
    unet = MaskDiT(**params['model']).to(device)
    unet.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    unet.eval()

    accelerator = Accelerator(mixed_precision="fp16")
    unet = accelerator.prepare(unet)

    # Load noise scheduler
    noise_scheduler = DDIMScheduler(**params['diff'])

    latents = torch.randn((1, 128, 128), device=device)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    return autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params


def generate_audio(text, autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params, device,
                   length=10,
                   guidance_scale=5, guidance_rescale=0.75, ddim_steps=100, eta=1,
                   random_seed=None, randomize_seed=False):
    neg_text = None
    length = length * params['autoencoder']['latent_sr']

    gt, gt_mask = None, None

    if text == '':
        guidance_scale = None
        print('empyt input')

    if randomize_seed:
        random_seed = random.randint(0, MAX_SEED)

    pred = inference(autoencoder, unet,
                     gt, gt_mask,
                     tokenizer, text_encoder,
                     params, noise_scheduler,
                     text, neg_text,
                     length,
                     guidance_scale, guidance_rescale,
                     ddim_steps, eta, random_seed,
                     device)

    pred = pred.cpu().numpy().squeeze(0).squeeze(0)
    # output_file = f"{save_path}/{text}.wav"
    # sf.write(output_file, pred, samplerate=params['autoencoder']['sr'])

    return params['autoencoder']['sr'], pred


def editing_audio(text, autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params, device,
                  boundary,
                  gt_file, mask_start, mask_length,
                  guidance_scale, guidance_rescale, ddim_steps, eta,
                  random_seed, randomize_seed):
    neg_text = None
    # max_length = 10

    if text == '':
        guidance_scale = None
        print('empyt input')

    mask_end = mask_start + mask_length

    # Load and preprocess ground truth audio
    gt, sr = librosa.load(gt_file, sr=params['autoencoder']['sr'])
    gt = gt / (np.max(np.abs(gt)) + 1e-9)

    audio_length = len(gt) / sr
    mask_start = min(mask_start, audio_length)
    if mask_end > audio_length:
        # outpadding mode
        padding = round((mask_end - audio_length)*params['autoencoder']['sr'])
        gt = np.pad(gt, (0, padding), 'constant')
        audio_length = len(gt) / sr

    output_audio = gt.copy()

    gt = torch.tensor(gt).unsqueeze(0).unsqueeze(1).to(device)
    boundary = min((mask_end - mask_start)/2, boundary)
    # print(boundary)

    # Calculate start and end indices
    start_idx = max(mask_start - boundary, 0)
    end_idx = min(mask_end + boundary, audio_length)
    # print(start_idx)
    # print(end_idx)

    mask_start -= start_idx
    mask_end -= start_idx

    gt = gt[:, :, round(start_idx*params['autoencoder']['sr']):round(end_idx*params['autoencoder']['sr'])]

    # Encode the audio to latent space
    gt_latent = autoencoder(audio=gt)
    B, D, L = gt_latent.shape
    length = L

    gt_mask = torch.zeros(B, D, L).to(device)
    latent_sr = params['autoencoder']['latent_sr']
    gt_mask[:, :, round(mask_start * latent_sr): round(mask_end * latent_sr)] = 1
    gt_mask = gt_mask.bool()

    if randomize_seed:
        random_seed = random.randint(0, MAX_SEED)

    # Perform inference to get the edited latent representation
    pred = inference(autoencoder, unet,
                     gt_latent, gt_mask,
                     tokenizer, text_encoder,
                     params, noise_scheduler,
                     text, neg_text,
                     length,
                     guidance_scale, guidance_rescale,
                     ddim_steps, eta, random_seed,
                     device)

    pred = pred.cpu().numpy().squeeze(0).squeeze(0)

    chunk_length = end_idx - start_idx
    pred = pred[:round(chunk_length*params['autoencoder']['sr'])]

    output_audio[round(start_idx*params['autoencoder']['sr']):round(end_idx*params['autoencoder']['sr'])] = pred

    pred = output_audio

    return params['autoencoder']['sr'], pred


MAX_SEED = np.iinfo(np.int32).max
