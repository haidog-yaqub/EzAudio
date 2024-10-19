from api.ezaudio import EzAudio
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

ezaudio = EzAudio(model_name='s3_xl', device='cpu')

if __name__ == '__main__':
    prompt = "a dog barking in the distance"
    sr, audio = ezaudio.generate_audio(prompt, ddim_steps=25)
    sf.write(f'{prompt}.wav', audio, sr)