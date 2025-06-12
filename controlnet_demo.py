from api.controlnet import EzAudio_ControlNet
import torch
import soundfile as sf
import pandas as pd
import os
from tqdm import tqdm


if __name__ == '__main__':
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    controlnet = EzAudio_ControlNet(model_name='energy', device=device)

    prompt = 'dog barking'
    audio_path = 'egs/reference.mp3'

    sr, audio = controlnet.generate_audio(prompt, audio_path=audio_path)
    sf.write(f"{prompt}.wav", audio, samplerate=sr)