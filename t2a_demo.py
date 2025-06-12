from api.ezaudio import EzAudio
import torch
import soundfile as sf
import pandas as pd
import os
from tqdm import tqdm


if __name__ == '__main__':
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ezaudio = EzAudio(model_name='s3_l', device=device)
   
    prompt = "a dog barking in the distance"
    sr, audio = ezaudio.generate_audio(prompt)
    sf.write(f'{prompt}.wav', audio, sr)