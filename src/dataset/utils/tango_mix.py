import torch
import torchaudio
import numpy as np
import soundfile as sf


def a_weight(fs, n_fft, min_db=-80.0):
    freq = torch.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = freq ** 2
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * torch.log10(torch.tensor(12194.0)) + 2 * torch.log10(freq_sq)
                           - torch.log10(freq_sq + 12194.0 ** 2)
                           - torch.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * torch.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * torch.log10(freq_sq + 737.9 ** 2))
    weight = torch.maximum(weight, torch.tensor(min_db))
    return weight


def compute_gain(sound, fs=24000, n_fft=1920, min_db=-80.0, mode="RMSE"):
    stride = n_fft // 2
    # Create overlapping windows
    windows = sound.unfold(0, n_fft, stride)

    if mode == "RMSE":
        gain = torch.mean(windows ** 2, dim=1)
    elif mode == "A_weighting":
        hann_window = torch.hann_window(n_fft)
        windows = hann_window * windows
        spec = torch.fft.rfft(windows)
        power_spec = torch.abs(spec) ** 2
        a_weighting = a_weight(fs, n_fft)
        a_weighted_spec = power_spec * torch.tensor(np.power(10, a_weighting / 10))
        gain = torch.sum(a_weighted_spec, dim=1)
    else:
        raise Exception("Invalid mode {}".format(mode))

    gain = torch.maximum(gain, torch.tensor(np.power(10, min_db / 10)))
    gain_db = 10 * torch.log10(gain)
    return gain_db


def tango_audio_mix(sound1, sound2, r=0.5, fs=24000, n_fft=1920):
    gain1 = torch.max(compute_gain(sound1[0], fs, n_fft))  # Decibel
    gain2 = torch.max(compute_gain(sound2[0], fs, n_fft))
    # print(gain1)
    # print(gain2)
    t = 1.0 / (1 + np.power(10., (gain1 - gain2) / 20.) * (1 - r) / r)
    # print(t)
    sound = ((sound1 * t + sound2 * (1 - t)) / torch.sqrt(t ** 2 + (1 - t) ** 2))
    return sound


if __name__ == '__main__':
    a1, sr = torchaudio.load('../../../debug_audio/01/1.wav')
    a2, sr = torchaudio.load('../../../debug_audio/01/4.wav')

    a1 = a1[:, :24000*4]
    a2 = a2[:, :24000*4]

    mixsound = tango_audio_mix(a1, a2, 0.5, 24000, 1920)

    sf.write('mixed.wav', mixsound.numpy(), 24000)