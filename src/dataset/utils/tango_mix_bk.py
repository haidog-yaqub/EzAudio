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


def compute_gain(sound, fs=24000, n_fft=1920, min_db=-80.0, mode="A_weighting"):
    # ref n_fft: 16000-2048, 44100-4096
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == "RMSE":
            g = torch.mean(sound[i: i + n_fft] ** 2)
        elif mode == "A_weighting":
            spec = torch.fft.rfft(torch.hann_window(n_fft) * sound[i: i + n_fft])
            power_spec = torch.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = torch.sum(a_weighted_spec)
        else:
            raise Exception("Invalid mode {}".format(mode))
        gain.append(g.item())

    gain = torch.tensor(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * torch.log10(gain)
    return gain_db


def tango_audio_mix(sound1, sound2, r=0.5, fs=24000, n_fft=1920):
    gain1 = torch.max(compute_gain(sound1, fs, n_fft))  # Decibel
    gain2 = torch.max(compute_gain(sound2, fs, n_fft))
    # print(gain1)
    # print(gain2)
    t = 1.0 / (1 + np.power(10., (gain1 - gain2) / 20.) * (1 - r) / r)
    # print(t)
    sound = ((sound1 * t + sound2 * (1 - t)) / torch.sqrt(t ** 2 + (1 - t) ** 2))
    return sound


if __name__ == '__main__':
    a1, sr = torchaudio.load('../../../debug_audio/01/1.wav')
    a2, sr = torchaudio.load('../../../debug_audio/01/4.wav')

    a1 = a1[0, :24000*4]
    a2 = a2[0, :24000*4]

    mixsound = tango_audio_mix(a1, a2, 0.5, 24000, 1920)

    sf.write('mixed.wav', mixsound.numpy(), 24000)