import torch
import torchaudio
import soundfile as sf
import random
import scipy
from .tango_mix import tango_audio_mix


class RandomAdd180Phase():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.mul_(-1)
        return sample


class RandomAmp():
    def __init__(self, low, high, p=0.5):
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            # fixed, waveform value would not exceed 1.0/-1.0
            max_val = sample.abs().max().item()
            if max_val != 0:
                max_possible_amp = min(1.0 / max_val, self.high)
            else:
                max_possible_amp = self.high
            amp = torch.FloatTensor(1).uniform_(self.low, max_possible_amp)
            sample.mul_(amp)
        return sample


class RandomMuLawCompression():
    # quatization
    def __init__(self, p=0.5, n_channels=256):
        self.n_channels = n_channels
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            e = torchaudio.functional.mu_law_encoding(sample, self.n_channels)
            sample = torchaudio.functional.mu_law_decoding(e, self.n_channels)
        return sample


class RandomLowPassFilter():
    # lp
    def __init__(self, fs, p=0.5, fc_threshold=(0.75, 1.00)):
        self.p = p
        self.fs = fs
        self.fc_threshold = fc_threshold

    def __call__(self, sample):
        if random.random() < self.p:
            print('low pass')
            fc = random.uniform(*self.fc_threshold)
            fc = fc * self.fs / 2
            sample = torchaudio.functional.lowpass_biquad(sample, self.fs, fc)
        return sample


class RandomHighPassFilter():
    # hp
    def __init__(self, fs, p=0.5, fc_threshold=(0.00, 0.25)):
        self.p = p
        self.fs = fs
        self.fc_threshold = fc_threshold

    def __call__(self, sample):
        if random.random() < self.p:
            print('high pass')
            fc = random.uniform(*self.fc_threshold)
            fc = fc * self.fs / 2
            sample = torchaudio.functional.highpass_biquad(sample, self.fs, fc)
        return sample


class WavAugmentation():
    def __init__(self, aug_config):
        pipeline = []
        for k in aug_config:
            if k == 'phase180':
                pipeline.append(RandomAdd180Phase(aug_config[k]['p']))
            elif k == 'amplitude':
                pipeline.append(RandomAmp(aug_config[k]['low'], aug_config[k]['high'], aug_config[k]['p']))
            elif k == 'mu_law':
                pipeline.append(RandomMuLawCompression(aug_config[k]['p'], aug_config[k]['n_channels']))
            elif k == 'low_pass':
                pipeline.append(RandomLowPassFilter(aug_config[k]['fs'], aug_config[k]['p'], aug_config[k]['fc_threshold']))
            elif k == 'high_pass':
                pipeline.append(RandomHighPassFilter(aug_config[k]['fs'], aug_config[k]['p'], aug_config[k]['fc_threshold']))
        self.pipeline = pipeline

    def __call__(self, sample):
        for transform in self.pipeline:
            sample = transform(sample)
        return sample


if __name__ == '__main__':
    a1, sr = torchaudio.load('../../../debug_audio/01/1.wav')
    a2, sr = torchaudio.load('../../../debug_audio/01/4.wav')

    a1 = a1[0, :24000*4]
    a2 = a2[0, :24000*4]

    # mixsound = tango_audio_mix(a1, a2, 0.5, 24000, 1920)
    filter = RandomLowPassFilter(24000, 1, (0.75, 1))
    wav = filter(a1)
    sf.write('mixed.wav', wav.numpy(), 24000)