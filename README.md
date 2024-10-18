<img src="ezaudio.png">

# EzAudio: Enhancing Text-to-Audio Generation with Efficient Diffusion Transformer

🟣 EzAudio is a diffusion-based text-to-audio generation model. Designed for real-world audio applications, EzAudio brings together high-quality audio synthesis with lower computational demands.

🎛 Play EzAudio on Hugging Face Space: [EzAudio: Text-to-Audio Generation, Editing, and Inpainting](https://huggingface.co/spaces/OpenSound/EzAudio) and [EzAudio-ControlNet](https://huggingface.co/spaces/OpenSound/EzAudio-ControlNet)!

## Installation

Clone the repository and install the dependencies:
```
git clone git@github.com:haidog-yaqub/EzAudio.git
cd EzAudio
pip install -r requirements.txt
```

## Usage

You can use the model with the following code:

```python
from api import EzAudio
```

## Todo
- [x] Release Gradio Demo along with checkpoints [EzAudio Space](https://huggingface.co/spaces/OpenSound/EzAudio)
- [x] Release ControlNet Demo along with checkpoints [EzAudio ControlNet Space](https://huggingface.co/spaces/OpenSound/EzAudio-ControlNet)
- [x] Release inference code 
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
