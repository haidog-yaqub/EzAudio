import torch
import torch.nn as nn
from .dac import DAC
from .stable_vae import load_vae


class Autoencoder(nn.Module):
    def __init__(self, ckpt_path, model_type='dac', quantization_first=False):
        super(Autoencoder, self).__init__()
        self.model_type = model_type
        if self.model_type == 'dac':
            model = DAC.load(ckpt_path)
        elif self.model_type == 'stable_vae':
            model = load_vae(ckpt_path)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")
        self.ae = model.eval()
        self.quantization_first = quantization_first
        print(f'Autoencoder quantization first mode: {quantization_first}')

    @torch.no_grad()
    def forward(self, audio=None, embedding=None):
        if self.model_type == 'dac':
            return self.process_dac(audio, embedding)
        elif self.model_type == 'encodec':
            return self.process_encodec(audio, embedding)
        elif self.model_type == 'stable_vae':
            return self.process_stable_vae(audio, embedding)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")

    def process_dac(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z, *_ = self.ae.quantizer(z, None)
            return z
        elif embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z, *_ = self.ae.quantizer(z, None)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

    def process_encodec(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                code = self.ae.quantizer.encode(z)
                z = self.ae.quantizer.decode(code)
            return z
        elif embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                code = self.ae.quantizer.encode(z)
                z = self.ae.quantizer.decode(code)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

    def process_stable_vae(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z = self.ae.bottleneck.encode(z)
            return z
        if embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z = self.ae.bottleneck.encode(z)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")
