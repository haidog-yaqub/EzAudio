import torch
import random
import numpy as np
import librosa
from accelerate import Accelerator
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler
from src.models.conditioners import MaskDiT
from src.models.controlnet import DiTControlNet
from src.models.conditions import Conditioner
from src.modules.autoencoder_wrapper import Autoencoder
from src.inference_controlnet import inference
from src.utils import load_yaml_with_includes


# Load model and configs
def load_models(config_name, ckpt_path, controlnet_path, vae_path, device):
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

    accelerator = Accelerator(mixed_precision="fp16")
    unet, controlnet = accelerator.prepare(unet, controlnet)

    # Load noise scheduler
    noise_scheduler = DDIMScheduler(**params['diff'])

    latents = torch.randn((1, 128, 128), device=device)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    return autoencoder, unet, controlnet, conditioner, tokenizer, text_encoder, noise_scheduler, params


MAX_SEED = np.iinfo(np.int32).max

# Model and config paths
config_name = 'ckpts/controlnet/energy_l.yml'
ckpt_path = 'ckpts/s3/ezaudio_s3_l.pt'
controlnet_path = 'ckpts/controlnet/s3_l_energy.pt'
vae_path = 'ckpts/vae/1m.pt'
# save_path = 'output/'
# os.makedirs(save_path, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

(autoencoder, unet, controlnet, conditioner, 
 tokenizer, text_encoder, noise_scheduler, params) = load_models(config_name, ckpt_path, controlnet_path, vae_path, device)


def generate_audio(text,
                   audio_path, surpass_noise,
                   guidance_scale, guidance_rescale, 
                   ddim_steps, eta,
                   conditioning_scale,
                   random_seed, randomize_seed):
    sr = params['autoencoder']['sr']

    gt, _ = librosa.load(audio_path, sr=sr)
    gt = gt / (np.max(np.abs(gt)) + 1e-9)  # Normalize audio

    if surpass_noise > 0:
        mask = np.abs(gt) <= surpass_noise
        gt[mask] = 0

    original_length = len(gt)
    # Ensure the audio is of the correct length by padding or trimming
    duration_seconds = min(len(gt) / sr, 10)
    quantized_duration = np.ceil(duration_seconds * 2) / 2  # This rounds to the nearest 0.5 seconds
    num_samples = int(quantized_duration * sr)
    audio_frames = round(num_samples / sr * params['autoencoder']['latent_sr'])

    if len(gt) < num_samples:
        padding = num_samples - len(gt)
        gt = np.pad(gt, (0, padding), 'constant')
    else:
        gt = gt[:num_samples]

    gt_audio = torch.tensor(gt).unsqueeze(0).unsqueeze(1).to(device)
    gt = autoencoder(audio=gt_audio)
    condition = conditioner(gt_audio.squeeze(1), gt.shape)

    # Handle random seed
    if randomize_seed:
        random_seed = random.randint(0, MAX_SEED)

    # Perform inference
    pred = inference(autoencoder, unet, controlnet,
                     None, None, condition,
                     tokenizer, text_encoder, 
                     params, noise_scheduler,
                     text, neg_text=None,
                     audio_frames=audio_frames, 
                     guidance_scale=guidance_scale, guidance_rescale=guidance_rescale, 
                     ddim_steps=ddim_steps, eta=eta, random_seed=random_seed, 
                     conditioning_scale=conditioning_scale, device=device)

    pred = pred.cpu().numpy().squeeze(0).squeeze(0)[:original_length]

    return sr, pred


examples_energy = [
    ["Dog barking in the background", "reference.mp3"],
    ["Duck quacking", "reference2.mp3"],
    ["Truck honking on the street", "reference3.mp3"]
]