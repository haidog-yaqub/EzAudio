import pandas as pd
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from .utils.wav_aug import WavAugmentation
import soundfile as sf


class EACaps(Dataset):
    def __init__(self, data_dir, meta_dir, subset='train', fine_tune=True,
                 seg_length=10, sr=24000,
                 aug_config=None,
                 norm=True, mono=True,
                 text_path=None, 
                 uncond_path=None, cfg_prob=0.0,
                 prepare_mode=False,
                 debug=False,
                 **kwargs):
        self.datadir = data_dir
        meta = pd.read_csv(meta_dir)
        meta = meta[meta['split'] == subset]
        if fine_tune:
            meta = meta[meta['fine_tune_data']==True]
        if prepare_mode is False:
            meta = meta[meta['audio_length'] != 0]
        self.meta = meta
        self.seg_len = seg_length
        self.sr = sr
        self.augmenter = WavAugmentation(aug_config) if aug_config is not None else nn.Identity()
        self.prepare_mode = prepare_mode
        if self.prepare_mode is False:
            self.text_path = text_path
            if self.text_path is not None:
                assert uncond_path is not None
                self.uncond_text = torch.load(uncond_path)
                self.cfg_prob = cfg_prob
        # this mode make sure all text embed
        self.norm = norm
        self.mono = mono
        self.debug = debug
        if self.debug:
            print('checking text embedding')
            self.cfg_prob = 0.0

    def load_audio(self, audio_path):
        y, sr = torchaudio.load(audio_path)
        assert sr == self.sr

        # Handle different channel configurations
        if y.shape[0] == 6:
            # Downmix six-channel audio to stereo by averaging front left and front right channels
            y = torch.mean(y[:2, :], dim=0, keepdim=True)
            
        # Handle stereo or mono based on self.mono
        if self.mono:
            # Convert to mono by averaging all channels
            y = torch.mean(y, dim=0, keepdim=True)
        else:
            if y.shape[0] == 1:
                pass
            elif y.shape[0] == 2:
                # Randomly pick one of the two stereo channels or take the mean
                if random.choice([True, False]):
                    y = torch.mean(y, dim=0, keepdim=True)
                else:
                    channel = random.choice([0, 1])
                    y = y[channel, :].unsqueeze(0)
            else:
                raise ValueError("Unsupported number of channels: {}".format(y.shape[0]))

        total_length = y.shape[-1]
        if int(total_length - self.sr * self.seg_len) > 0:
            start = np.random.randint(0, int(total_length - self.sr * self.seg_len) + 1)
        else:
            start = 0
        end = min(start + self.seg_len * self.sr, total_length)

        audio_clip = torch.zeros(self.seg_len * self.sr)
        audio_clip[:end - start] = y[0, start:end]
        
        if self.norm:
            eps = 1e-9
            max_val = torch.max(torch.abs(audio_clip))
            audio_clip = audio_clip / (max_val + eps)
        
        audio_clip = self.augmenter(audio_clip)
        return audio_clip.unsqueeze(0)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        text = row['caption']
        # load current audio
        audio_path = self.datadir + str(row['audio_path'])
        if self.prepare_mode:
            return text, str(row['absolute_index'])

        if self.debug:
            audio_clip = 0
        else:
            audio_clip = self.load_audio(audio_path)

        if self.text_path:
            if torch.rand(1) < self.cfg_prob:
                text_embed = self.uncond_text
            else:
                # text_path = os.path.join(self.text_path, str(row['audio_path']).replace('.wav', '.pt'))
                text_path = os.path.join(self.text_path, str(row['absolute_index'])+'.pt')
                text_embed = torch.load(text_path)
            return audio_clip, text_embed['embedding'], text_embed['mask']
        else:
            return audio_clip, text

    def __len__(self):
        return len(self.meta)
    #     break