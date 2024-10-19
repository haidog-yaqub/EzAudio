<img src="arts/ezaudio.png">

# EzAudio: Enhancing Text-to-Audio Generation with Efficient Diffusion Transformer

🟣 EzAudio is a diffusion-based text-to-audio generation model. Designed for real-world audio applications, EzAudio brings together high-quality audio synthesis with lower computational demands.

🎛 Play with EzAudio for text-to-audio generation, editing, and inpainting: [EzAudio](https://huggingface.co/spaces/OpenSound/EzAudio)

🎮 EzAudio-ControlNet is available: [EzAudio-ControlNet](https://huggingface.co/spaces/OpenSound/EzAudio-ControlNet)

We want to thank Hugging Face Space and Gradio for providing incredible demo platform.

## Installation

Clone the repository:
```
git clone git@github.com:haidog-yaqub/EzAudio.git
```
Install the dependencies:
```
cd EzAudio
pip install -r requirements.txt
```
Download checkponts from: [https://huggingface.co/OpenSound/EzAudio](https://huggingface.co/OpenSound/EzAudio/tree/main)

## Usage

You can use the model with the following code:

```python
from api.ezaudio import EzAudio
import torch
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ezaudio = EzAudio(model_name='s3_xl', device='cpu')

prompt = "a dog barking in the distance"
sr, audio = ezaudio.generate_audio(prompt)
sf.write(f'{prompt}.wav', audio, sr)
```

## Todo
- [x] Release Gradio Demo along with checkpoints [EzAudio Space](https://huggingface.co/spaces/OpenSound/EzAudio)
- [x] Release ControlNet Demo along with checkpoints [EzAudio ControlNet Space](https://huggingface.co/spaces/OpenSound/EzAudio-ControlNet)
- [x] Release inference code
- [ ] Improve API and support automatic ckpts downloading [WIP]
- [ ] Release checkpoints for stage1 and stage2
- [ ] Release training pipeline and dataset

## Reference

If you find the code useful for your research, please consider citing:

```bibtex
@article{hai2024ezaudio,
  title={EzAudio: Enhancing Text-to-Audio Generation with Efficient Diffusion Transformer},
  author={Hai, Jiarui and Xu, Yong and Zhang, Hao and Li, Chenxing and Wang, Helin and Elhilali, Mounya and Yu, Dong},
  journal={arXiv preprint arXiv:2409.10819},
  year={2024}
}
```

## Acknowledgement
Some code are borrowed from or inspired by: [U-Vit](https://github.com/baofff/U-ViT), [Pixel-Art](https://github.com/PixArt-alpha/PixArt-alpha), [Huyuan-DiT](https://github.com/Tencent/HunyuanDiT), and [Stable Audio](https://github.com/Stability-AI/stable-audio-tools).
