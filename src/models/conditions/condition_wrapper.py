import torch
import torch.nn as nn
from .chroma import ChromaExtractor
from .energy import EnergyExtractor
from .voice import VoiceConversionExtractor
from .mbenergy import MultibandEnergyExtractor


class Conditioner(nn.Module):
    def __init__(self,
                 condition_type,
                 **kwargs
                ):
        super().__init__()
        if condition_type == 'energy':
            self.conditioner = EnergyExtractor(**kwargs)
        elif condition_type == 'chroma':
            self.conditioner = ChromaExtractor(**kwargs)
        elif condition_type == 'vc':
            self.conditioner = VoiceConversionExtractor(**kwargs)
        elif condition_type == 'mb_energy':
            self.conditioner = MultibandEnergyExtractor(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, waveform, latent_shape):
        # B T C
        condition = self.conditioner(waveform)
        # B C T
        condition = condition.permute(0, 2, 1).contiguous()

        if len(latent_shape) == 4:
            # 2d spectrogram B C T F
            assert (condition.shape[-1] % latent_shape[-2]) == 0
            X = latent_shape[-1] * condition.shape[-1] // latent_shape[-2]
            # copy on F direction
            condition = condition.unsqueeze(-1).expand(-1, -1, -1, X)
        elif len(latent_shape) == 3:
            condition = condition
        else:
            raise NotImplementedError
        return condition


if __name__ == '__main__':
    conditioner = Conditioner(condition_type='energy',
                              hop_size=160, window_size=1024, padding='reflect',
                              min_db=-80, norm=True)
    audio = torch.rand(4, 16000)  # Example audio signal
    energy = conditioner(audio, (4, 8, 100, 64))