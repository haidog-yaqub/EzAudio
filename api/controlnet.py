import os
import sys
import torch
import random
import numpy as np
import librosa
import urllib.request
from pathlib import Path
from accelerate import Accelerator
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler
from src.models.conditioners import MaskDiT
from src.models.controlnet import DiTControlNet
from src.models.conditions import Conditioner
from src.modules.autoencoder_wrapper import Autoencoder
from src.inference_controlnet import inference
from src.utils import load_yaml_with_includes


configs = {'model': {'path': 'ckpts/s3/ezaudio_s3_l.pt',
                     'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/s3/ezaudio_s3_l.pt'},
           'energy': {'path': 'ckpts/controlnet/s3_l_energy.pt',
                      'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/controlnet/s3_l_energy.pt',
                      'config': 'ckpts/controlnet/energy_l.yml'},
           'vae': {'path': 'ckpts/vae/1m.pt',
                   'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/vae/1m.pt'}
          }
MAX_SEED = np.iinfo(np.int32).max


class EzAudio_ControlNet:
    def __init__(self, model_name, ckpt_path=None, controlnet_path=None, vae_path=None, device='cuda'):
        self.device = device
        config_name = configs[model_name]['config']
        if ckpt_path is None:
            ckpt_path = self.download_ckpt(configs['model'])

        if controlnet_path is None:
            controlnet_path = self.download_ckpt(configs[model_name])

        if vae_path is None:
            vae_path = self.download_ckpt(configs['vae'])

        (self.autoencoder, self.unet, self.controlnet,
         self.conditioner, self.tokenizer,
         self.text_encoder, self.noise_scheduler,
         self.params) = self.load_models(config_name, ckpt_path, controlnet_path,
                                         vae_path, device)

    def download_ckpt(self, model_dict):
        local_path = Path(model_dict['path'])
        url = model_dict['url']
        # Create directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists() and url:
            print(f"Downloading from {url} to {local_path}...")

            def progress_bar(block_num, block_size, total_size):
                downloaded = block_num * block_size
                progress = downloaded / total_size * 100
                sys.stdout.write(f"\rProgress: {progress:.2f}%")
                sys.stdout.flush()
            try:
                urllib.request.urlretrieve(url, local_path, reporthook=progress_bar)
                print(f"Downloaded checkpoint to {local_path}")
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
        else:
            print(f"Checkpoint already exists at {local_path}")
        return local_path

    # Load model and configs
    def load_models(self, config_name, ckpt_path, controlnet_path, vae_path, device):
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

        controlnet_config = params['model'].copy()
        controlnet_config.update(params['controlnet'])
        controlnet = DiTControlNet(**controlnet_config).to(device)
        controlnet.eval()
        controlnet.load_state_dict(torch.load(controlnet_path, map_location='cpu')['model'])
        conditioner = Conditioner(**params['conditioner']).to(device)

        # accelerator = Accelerator(mixed_precision="fp16")
        # unet, controlnet = accelerator.prepare(unet, controlnet)

        # Load noise scheduler
        noise_scheduler = DDIMScheduler(**params['diff'])

        latents = torch.randn((1, 128, 128), device=device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
        _ = noise_scheduler.add_noise(latents, noise, timesteps)

        return autoencoder, unet, controlnet, conditioner, tokenizer, text_encoder, noise_scheduler, params

    def generate_audio(self, text,
                       audio_path, surpass_noise=0,
                       guidance_scale=3.5, guidance_rescale=0, 
                       ddim_steps=50, eta=1,
                       conditioning_scale=1,
                       random_seed=None, randomize_seed=False):

        sr = self.params['autoencoder']['sr']

        gt, _ = librosa.load(audio_path, sr=sr)
        gt = gt / (np.max(np.abs(gt)) + 1e-9)  # Normalize audio

        if surpass_noise > 0:
            mask = np.abs(gt) <= surpass_noise
            gt[mask] = 0

        original_length = len(gt)
        # Ensure the audio is of the correct length by padding or trimming
        num_samples = int(10 * sr)
        audio_frames = round(num_samples / sr * self.params['autoencoder']['latent_sr'])

        if len(gt) < num_samples:
            padding = num_samples - len(gt)
            gt = np.pad(gt, (0, padding), 'constant')
        else:
            gt = gt[:num_samples]

        gt_audio = torch.tensor(gt).unsqueeze(0).unsqueeze(1).to(self.device)
        gt = self.autoencoder(audio=gt_audio)
        condition = self.conditioner(gt_audio.squeeze(1), gt.shape)

        # Handle random seed
        if randomize_seed:
            random_seed = random.randint(0, MAX_SEED)

        # Perform inference
        pred = inference(self.autoencoder, self.unet, self.controlnet,
                         None, None, condition,
                         self.tokenizer, self.text_encoder, 
                         self.params, self.noise_scheduler,
                         text, neg_text=None,
                         audio_frames=audio_frames, 
                         guidance_scale=guidance_scale, guidance_rescale=guidance_rescale, 
                         ddim_steps=ddim_steps, eta=eta, random_seed=random_seed, 
                         conditioning_scale=conditioning_scale, device=self.device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)[:original_length]

        return sr, pred