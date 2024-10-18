from transformers import HubertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
import librosa


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class VoiceConversionExtractor(nn.Module):
    # training on the fly might be slow
    def __init__(self, config, sr):
        super().__init__()
        self.encoder = HubertModelWithFinalProj.from_pretrained(config)
        self.encoder.eval()
        self.sr = sr
        self.target_sr = 16000
        if self.sr != self.target_sr:
            self.resampler = torchaudio.transforms.Resample(orig_freq=self.sr,
                                                            new_freq=self.target_sr)

    def forward(self, audio):
        if self.sr != self.target_sr:
            audio = self.resampler(audio)
        audio = F.pad(audio, ((400 - 320) // 2, (400 - 320) // 2))
        logits = self.encoder(audio)['last_hidden_state']
        return logits


if __name__ == '__main__':
    model = VoiceConversionExtractor('lengyue233/content-vec-best', 24000)
    audio, sr = librosa.load('test.wav', sr=24000)
    audio = audio[:round(100*320*1.5)]
    audio = torch.tensor([audio])
    with torch.no_grad():
        content = model(audio)
    print(content.shape)