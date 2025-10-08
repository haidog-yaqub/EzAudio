from api.ezaudio import EzAudio
import torch
import soundfile as sf

if __name__ == '__main__':
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ezaudio = EzAudio(model_name='s3_xl', device=device)
    
    # text to audio genertation
    prompt = "a dog barking in the distance"
    sr, audio = ezaudio.generate_audio(prompt)
    sf.write(f'{prompt}.wav', audio, sr)
    
    # audio inpainting
    prompt = "A train passes by, blowing its horns"
    original_audio = 'egs/edit_example.wav'
    sr, audio = ezaudio.editing_audio(prompt, boundary=2, gt_file=original_audio,
                                      mask_start=1, mask_length=5)
    sf.write(f'{prompt}_edit.wav', audio, sr)