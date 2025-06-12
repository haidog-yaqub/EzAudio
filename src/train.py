import random
import argparse
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import DDIMScheduler

from models.udit import RMSNorm
from models.conditioners import MaskDiT
from modules.autoencoder_wrapper import Autoencoder
from transformers import T5Tokenizer, T5EncoderModel
from dataset.audiocaps_v2 import EACaps
from utils import scale_shift, get_lr_scheduler, compute_snr, load_yaml_with_includes

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()

    # Config settings
    parser.add_argument('--config-name', type=str, default='config/ezaudio-l.yml')

    # Training settings
    parser.add_argument("--amp", type=str, default='fp16')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--save-every-step', type=int, default=5000)

    # Log and random seed
    parser.add_argument('--random-seed', type=int, default=2024)
    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='../logs/')
    parser.add_argument('--save-dir', type=str, default='../ckpts/')

    # fine-tune settings
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--strict', type=bool, default=False)
    return parser.parse_args()


def setup_directories(args, params):
    args.log_dir = os.path.join(args.log_dir, params['model_name']) + '/'
    args.save_dir = os.path.join(args.save_dir, params['model_name']) + '/'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)


def set_device(args):
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'


def setup_optimizer(unet, params):
    decay = set()
    no_decay = set()

    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, RMSNorm)
    no_decay_suffixes = ['bias', 'abs_pe', 'alpha', 'beta', 
                         'mask_embed',
                         'scale_shift_table', 'cfg_embedding']

    for mn, m in unet.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            if any(pn.endswith(suffix) for suffix in no_decay_suffixes):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in unet.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))],
         "weight_decay": params['opt']['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 
         "weight_decay": 0.0},]

    optimizer = torch.optim.AdamW(optim_groups, lr=params['opt']['learning_rate'], 
                                  betas=(params['opt']['beta1'], params['opt']['beta2']), eps=params['opt']['adam_epsilon'])
    return optimizer


def prepare_batch(args, batch, autoencoder, tokenizer, text_encoder, params):
    audio_clip, text_batch = batch
    with torch.no_grad():
        audio_clip = autoencoder(audio=audio_clip)
        if tokenizer is not None:
            # apply uncondition during training
            text_batch_np = np.array(text_batch)
            cfg_mask = torch.rand(len(text_batch_np)) < params['text_encoder']['cfg']
            text_batch_np[cfg_mask] = ""
            text_batch = text_batch_np.tolist()
            text_batch = tokenizer(text_batch, max_length=params['text_encoder']['max_length'], 
                                   padding="max_length", truncation=True, return_tensors="pt")
            text_mask = text_batch.attention_mask.to(audio_clip.device).bool()
            text = text_encoder(input_ids=text_batch.input_ids.to(audio_clip.device),
                                attention_mask=text_mask).last_hidden_state
        else:
            text, text_mask = None, None

    return audio_clip, text, text_mask


def prepare_batch_cache(args, batch, autoencoder, tokenizer, text_encoder, params):
    audio_clip, text, text_mask = batch
    with torch.no_grad():
        audio_clip = autoencoder(audio=audio_clip)
    if tokenizer is None:
        text, text_mask = None, None
    return audio_clip, text, text_mask


def compute_loss(model_pred, target, mask,
                 noise_scheduler, timesteps, snr_gamma=None):
    if snr_gamma is None:
        # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss * mask.float()
        loss = loss.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
        loss = loss.mean()
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]

        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        else:
            raise NotImplementedError

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss * mask.float()
        loss = loss.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
        loss = loss * mse_loss_weights
        loss = loss.mean()

    return loss


if __name__ == '__main__':
    # Fix the random seed
    args = parse_args()
    params = load_yaml_with_includes(args.config_name)
    args.stage = 'audioset' if params['model']['context_dim'] is None else 'audiocaps'
    if args.stage == 'audioset':
        guidance_scale = None
        args.mae = True
    else:
        guidance_scale = 3.0
        args.mae = False
    set_device(args)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp,
                              gradient_accumulation_steps=params['opt']['accumulation_steps'])

    train_set = EACaps(**params['data']['train'])
    if train_set.text_path is not None:
        args.offline = True
        print('offline text embedding')
        t5_device = 'cpu'
    else:
        args.offline = False
        t5_device = accelerator.device

    train_loader = DataLoader(train_set, num_workers=args.num_workers,
                              batch_size=params['opt']['batch_size'], shuffle=True)

    # Codec Model
    autoencoder = Autoencoder(ckpt_path=params['autoencoder']['path'],
                              model_type=params['autoencoder']['name'],
                              quantization_first=params['autoencoder']['q_first'])
    autoencoder.to(accelerator.device)
    autoencoder.eval()

    # text encoder
    if args.stage == 'audioset':
        tokenizer = None
        text_encoder = None
    elif args.stage == 'audiocaps':
        tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'])
        text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model'],
                                                      device_map='cpu').to(t5_device)
        text_encoder.eval()

    # main U-Net
    unet = MaskDiT(**params['model']).to(accelerator.device)

    # Load the state dict from pretrained model
    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location='cpu')['model']
    # Load the state dict into the model
    result = unet.load_state_dict(state_dict, strict=args.strict)

    if accelerator.is_main_process:
        # Check for missing keys and unexpected keys
        if result.missing_keys:
            print("Warning: The following layers were not loaded because they are missing in the checkpoint:")
            for key in result.missing_keys:
                print(f" - {key}")
        if result.unexpected_keys:
            print("Warning: The following layers were not expected in the model and thus were not loaded:")
            for key in result.unexpected_keys:
                print(f" - {key}")   
        total_params = sum([param.nelement() for param in unet.parameters()])
        print("Number of parameter: %.2fM" % (total_params / 1e6))
    accelerator.wait_for_everyone()

    noise_scheduler = DDIMScheduler(**params['diff'])

    optimizer = setup_optimizer(unet, params)
    lr_scheduler = get_lr_scheduler(optimizer, 'customized',
                                    warmup_steps=params['opt']['warmup'])
    # loss_func = torch.nn.MSELoss()

    unet, autoencoder, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        unet, autoencoder, optimizer, lr_scheduler, train_loader)

    global_step = 0.0
    losses = 0.0

    if accelerator.is_main_process:
        setup_directories(args, params)
        print(args)
    accelerator.wait_for_everyone()

    for epoch in range(args.epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_loader)):
            with accelerator.accumulate(unet):
                if args.offline:
                    (audio_clip,
                     text, text_mask) = prepare_batch_cache(args, batch,
                                                            autoencoder, tokenizer, text_encoder,
                                                            params)
                else:
                    (audio_clip,
                     text, text_mask) = prepare_batch(args, batch,
                                                      autoencoder, tokenizer, text_encoder,
                                                      params)
                # prepare training data (normalize and chunk length)
                audio_clip = scale_shift(audio_clip, params['autoencoder']['scale'],
                                         params['autoencoder']['shift'])
                audio_clip = audio_clip[:, :, :params['data']['train_frames']]

                # adding noise
                noise = torch.randn(audio_clip.shape).to(accelerator.device)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (noise.shape[0],),
                                          device=accelerator.device, ).long()
                noisy_target = noise_scheduler.add_noise(audio_clip, noise, timesteps)

                # output target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    velocity = noise_scheduler.get_velocity(audio_clip, noise, timesteps)
                    target = velocity

                # inference
                pred, mask = unet(noisy_target, timesteps, text, 
                                  context_mask=text_mask, cls_token=None, 
                                  gt=audio_clip)

                # backward
                loss = compute_loss(pred, target, mask, noise_scheduler, timesteps,
                                    snr_gamma=params['opt']['snr_gamma'])

                accelerator.backward(loss)
                # clip grad up sync step when using accumulation
                if accelerator.sync_gradients:
                    if 'grad_clip' in params['opt'] and params['opt']['grad_clip'] > 0:
                        accelerator.clip_grad_norm_(unet.parameters(),
                                                    max_norm=params['opt']['grad_clip'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1/params['opt']['accumulation_steps']
                losses += loss.item()/params['opt']['accumulation_steps']

            if accelerator.is_main_process:
                if global_step % args.log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    epoch_info = f'Epoch: [{epoch + 1}][{args.epochs}]'
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / args.log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    log_message = f'{current_time}\n{epoch_info}    {batch_info}    {loss_info}    {lr_info}\n'

                    with open(args.log_dir + 'log.txt', mode='a') as n:
                        n.write(log_message)

                    losses = 0.0

            if (global_step + 1) % args.save_every_step == 0:
                if accelerator.is_main_process:
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    accelerator.save({
                        "model": unwrapped_unet.state_dict(),
                    }, args.save_dir + str(global_step+1) + '.pt')
                    accelerator.save_state(f"{args.save_dir}{global_step + 1}")
                accelerator.wait_for_everyone()
                unet.train()

