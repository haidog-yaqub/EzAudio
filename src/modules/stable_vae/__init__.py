from .models.autoencoders import create_autoencoder_from_config
import os
import json
import torch
from torch.nn.utils import remove_weight_norm


def remove_all_weight_norm(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight_g'):
            remove_weight_norm(module)


def load_vae(ckpt_path, remove_weight_norm=False):
    config_file = os.path.join(os.path.dirname(ckpt_path), 'config.json')

    # Load the model configuration
    with open(config_file) as f:
        model_config = json.load(f)

    # Create the model from the configuration
    model = create_autoencoder_from_config(model_config)

    # Load the state dictionary from the checkpoint
    model_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

    # Strip the "autoencoder." prefix from the keys
    model_dict = {key[len("autoencoder."):]: value for key, value in model_dict.items() if key.startswith("autoencoder.")}

    # Load the state dictionary into the model
    model.load_state_dict(model_dict)

    # Remove weight normalization
    if remove_weight_norm:
        remove_all_weight_norm(model)

    # Set the model to evaluation mode
    model.eval()

    return model
