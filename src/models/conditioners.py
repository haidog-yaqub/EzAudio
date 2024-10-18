import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from .udit import UDiT
from .utils.span_mask import compute_mask_indices


class EmbeddingCFG(nn.Module):
    """
    Handles label dropout for classifier-free guidance.
    """
    # todo: support 2D input

    def __init__(self, in_channels):
        super().__init__()
        self.cfg_embedding = nn.Parameter(
            torch.randn(in_channels) / in_channels ** 0.5)

    def token_drop(self, condition, condition_mask, cfg_prob):
        """
        Drops labels to enable classifier-free guidance.
        """
        b, t, device = condition.shape[0], condition.shape[1], condition.device
        drop_ids = torch.rand(b, device=device) < cfg_prob
        uncond = repeat(self.cfg_embedding, "c -> b t c", b=b, t=t)
        condition = torch.where(drop_ids[:, None, None], uncond, condition)
        if condition_mask is not None:
            condition_mask[drop_ids] = False
            condition_mask[drop_ids, 0] = True

        return condition, condition_mask

    def forward(self, condition, condition_mask, cfg_prob=0.0):
        if condition_mask is not None:
            condition_mask = condition_mask.clone()
        if cfg_prob > 0:
            condition, condition_mask = self.token_drop(condition,
                                                        condition_mask,
                                                        cfg_prob)
        return condition, condition_mask


class DiscreteCFG(nn.Module):
    def __init__(self, replace_id=2):
        super(DiscreteCFG, self).__init__()
        self.replace_id = replace_id

    def forward(self, context, context_mask, cfg_prob):
        context = context.clone()
        if context_mask is not None:
            context_mask = context_mask.clone()
        if cfg_prob > 0:
            cfg_mask = torch.rand(len(context)) < cfg_prob
            if torch.any(cfg_mask):
                context[cfg_mask] = 0
                context[cfg_mask, 0] = self.replace_id
                if context_mask is not None:
                    context_mask[cfg_mask] = False
                    context_mask[cfg_mask, 0] = True
        return context, context_mask


class CFGModel(nn.Module):
    def __init__(self, context_dim, backbone):
        super().__init__()
        self.model = backbone
        self.context_cfg = EmbeddingCFG(context_dim)

    def forward(self, x, timesteps,
                context, x_mask=None, context_mask=None,
                cfg_prob=0.0):
        context = self.context_cfg(context, cfg_prob)
        x = self.model(x=x, timesteps=timesteps,
                       context=context,
                       x_mask=x_mask, context_mask=context_mask)
        return x


class ConcatModel(nn.Module):
    def __init__(self, backbone, in_dim, stride=[]):
        super().__init__()
        self.model = backbone

        self.downsample_layers = nn.ModuleList()
        for i, s in enumerate(stride):
            downsample_layer = nn.Conv1d(
                in_dim,
                in_dim * 2,
                kernel_size=2 * s,
                stride=s,
                padding=math.ceil(s / 2),
            )
            self.downsample_layers.append(downsample_layer)
            in_dim = in_dim * 2

        self.context_cfg = EmbeddingCFG(in_dim)

    def forward(self, x, timesteps,
                context, x_mask=None,
                cfg=False, cfg_prob=0.0):

        # todo: support 2D input
        # x: B, C, L
        # context: B, C, L

        for downsample_layer in self.downsample_layers:
            context = downsample_layer(context)

        context = context.transpose(1, 2)
        context = self.context_cfg(caption=context,
                                   cfg=cfg, cfg_prob=cfg_prob)
        context = context.transpose(1, 2)

        assert context.shape[-1] == x.shape[-1]
        x = torch.cat([context, x], dim=1)
        x = self.model(x=x, timesteps=timesteps,
                       context=None, x_mask=x_mask, context_mask=None)
        return x


class MaskDiT(nn.Module):
    def __init__(self, mae=False, mae_prob=0.5, mask_ratio=[0.25, 1.0], mask_span=10, **kwargs):
        super().__init__()
        self.model = UDiT(**kwargs)
        self.mae = mae
        if self.mae:
            out_channel = kwargs.pop('out_chans', None)
            self.mask_embed = nn.Parameter(torch.zeros((out_channel)))
            self.mae_prob = mae_prob
            self.mask_ratio = mask_ratio
            self.mask_span = mask_span

    def random_masking(self, gt, mask_ratios, mae_mask_infer=None):
        B, D, L = gt.shape
        if mae_mask_infer is None:
            # mask = torch.rand(B, L).to(gt.device) < mask_ratios.unsqueeze(1)
            mask_ratios = mask_ratios.cpu().numpy()
            mask = compute_mask_indices(shape=[B, L],
                                        padding_mask=None,
                                        mask_prob=mask_ratios,
                                        mask_length=self.mask_span,
                                        mask_type="static",
                                        mask_other=0.0,
                                        min_masks=1,
                                        no_overlap=False,
                                        min_space=0,)
            mask = mask.unsqueeze(1).expand_as(gt)
        else:
            mask = mae_mask_infer
            mask = mask.expand_as(gt)
        gt[mask] = self.mask_embed.view(1, D, 1).expand_as(gt)[mask]
        return gt, mask.type_as(gt)

    def forward(self, x, timesteps, context,
                x_mask=None, context_mask=None, cls_token=None,
                gt=None, mae_mask_infer=None,
                forward_model=True):
        # todo: handle controlnet inside
        mae_mask = torch.ones_like(x)
        if self.mae:
            if gt is not None:
                B, D, L = gt.shape
                mask_ratios = torch.FloatTensor(B).uniform_(*self.mask_ratio).to(gt.device)
                gt, mae_mask = self.random_masking(gt, mask_ratios, mae_mask_infer)
                # apply mae only to the selected batches
                if mae_mask_infer is None:
                    # determine mae batch
                    mae_batch = torch.rand(B) < self.mae_prob
                    gt[~mae_batch] = self.mask_embed.view(1, D, 1).expand_as(gt)[~mae_batch]
                    mae_mask[~mae_batch] = 1.0
            else:
                B, D, L = x.shape
                gt = self.mask_embed.view(1, D, 1).expand_as(x)
            x = torch.cat([x, gt, mae_mask[:, 0:1, :]], dim=1)

        if forward_model:
            x = self.model(x=x, timesteps=timesteps, context=context,
                           x_mask=x_mask, context_mask=context_mask,
                           cls_token=cls_token)
            # print(mae_mask[:, 0, :].sum(dim=-1))
        return x, mae_mask
