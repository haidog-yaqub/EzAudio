from api.ezaudio import load_models, generate_audio
import soundfile as sf
import torch


# Examples (if needed for the demo)
examples = [
    "a dog barking in the distance",
    "light guitar music is playing",
    "a duck quacks as waves crash gently on the shore",
    "footsteps crunch on the forest floor as crickets chirp",
    "a horse clip-clops in a windy rain as thunder cracks in the distance",
]

# Examples (if needed for the demo)
examples_edit = [
    ["A train passes by, blowing its horns", 2, 3],
    ["kids playing and laughing nearby", 5, 4],
    ["rock music playing on the street", 8, 6]
]

# Model and config paths
config_name = 'ckpts/ezaudio-xl.yml'
ckpt_path = 'ckpts/s3/ezaudio_s3_xl.pt'
vae_path = 'ckpts/vae/1m.pt'
# save_path = 'output/'
# os.makedirs(save_path, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

(autoencoder, unet, tokenizer,
 text_encoder, noise_scheduler, params) = load_models(config_name, ckpt_path,
                                                      vae_path, device)

if __name__ == '__main__':
    prompt = "a dog barking in the distance"
    sr, audio = generate_audio(prompt, autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params, device)
    sf.write(f'{prompt}.wav', audio, sr)