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

‘’python
from api import EzAudio
''

## Todo
- [x] Release Gradio Demo along with checkpoints [EzAudio Space](https://huggingface.co/spaces/OpenSound/EzAudio)
- [x] Release ControlNet Demo along with checkpoints [EzAudio ControlNet Space](https://huggingface.co/spaces/OpenSound/EzAudio-ControlNet)
- [x] Release inference code 
- [ ] Release checkpoints for stage1 and stage2
- [ ] Release training pipeline and dataset
