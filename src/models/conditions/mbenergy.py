import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import julius
import soundfile as sf


class MultibandEnergyExtractor(nn.Module):
    def __init__(self, hop_size: int = 512, window_size: int = 1024,
                 padding: str = 'reflect', min_db: float = -60,
                 norm: bool = True, quantize_levels: int = None,
                 n_bands: int = 8, control_bands: int = 4,
                 sample_rate: int = 24000,):
        super().__init__()
        self.hop_size = hop_size
        self.window_size = window_size
        self.padding = padding
        self.min_db = min_db
        self.norm = norm
        self.quantize_levels = quantize_levels
        self.n_bands = n_bands
        self.control_bands = control_bands
        self.sample_rate = sample_rate

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Split the audio into frequency bands
        audio = julius.split_bands(audio, n_bands=self.n_bands,
                                   sample_rate=self.sample_rate)[:self.control_bands].transpose(0, 1)
        B, C, _ = audio.shape
        for i in range(C):
            sf.write(f'output_{i}.wav', audio[0][i], self.sample_rate)

        # Compute number of frames
        n_frames = int(audio.size(-1) // self.hop_size)

        # Pad the audio signal
        pad_amount = (self.window_size - self.hop_size) // 2
        audio_padded = F.pad(audio, (pad_amount, pad_amount), mode=self.padding)

        # Square the padded audio signal
        audio_squared = audio_padded ** 2

        # Compute the mean energy for each frame using unfold and mean
        energy = audio_squared.unfold(dimension=-1, size=self.window_size, step=self.hop_size)
        energy = energy[:, :, :n_frames]
        print(energy.shape)
        energy = energy.mean(dim=-1)
        print(energy.shape)

        # Compute the square root of the mean energy to get the RMS energy
        # energy = torch.sqrt(energy)

        # Normalize the energy using the min_db value
        gain = torch.maximum(energy, torch.tensor(np.power(10, self.min_db / 10), device=audio.device))
        gain_db = 10 * torch.log10(gain)

        if self.norm:
            # Find the min and max of gain_db
            # min_gain_db = torch.min(gain_db)
            min_gain_db = self.min_db
            max_gain_db = torch.amax(gain_db, dim=(-1, -2), keepdim=True)

            # Avoid numerical error by adding a small epsilon to the denominator
            epsilon = 1e-8
            gain_db = (gain_db - min_gain_db) / (max_gain_db - min_gain_db + epsilon)

        if self.quantize_levels is not None:
            # Quantize the result to the given number of levels
            gain_db = torch.round(gain_db * (self.quantize_levels - 1)) / (self.quantize_levels - 1)

        return gain_db.transpose(-1, -2)


if __name__ == "__main__":
    energy_extractor = MultibandEnergyExtractor(hop_size=320, window_size=1280,
                                                padding='reflect',
                                                min_db=-60, norm=True)
    audio = torch.rand(4, 24000)
    energy = energy_extractor(audio)
    print(energy.shape)
    import librosa
    import matplotlib.pyplot as plt
    a1, _ = librosa.load('eg2.wav', sr=24000)
    audio = torch.tensor(a1[:5*16000]).unsqueeze(0)
    energy = energy_extractor(audio)
    print(energy.shape)

    # Plot the energy for each audio sample
    plt.figure(figsize=(12, 6))

    for i in range(energy.shape[-1]):
        plt.plot(energy[0, :, i].cpu().numpy(), label=f'Band {i+1}')

    plt.xlabel('Frame')
    plt.ylabel('Energy (dB)')
    plt.title('Energy over Time')
    plt.legend()
    plt.savefig('debug.png')
