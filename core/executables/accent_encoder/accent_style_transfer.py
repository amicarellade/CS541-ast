"""
Neural Style Transfer for Audio

Used https://pytorch.org/tutorials/advanced/neural_style_tutorial.html as a starting point

"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from models.accent_encoder import *
from utils.fileutils import *
from utils.oputils import *
from utils.loggingutils import *


""" ------ Neural Style Transfer ------ """


def build_vgg_transfer_model(pretrained_vgg: VGG19Net, 
                             content_spec: torch.Tensor, 
                             style_spec: torch.Tensor,
                             content_layers: list,
                             style_layers: list, 
                             device: torch.device) -> tuple[nn.Sequential, list[nn.Module], list[nn.Module]]:

    content_losses = list()
    style_losses = list()

    model = nn.Sequential(VGGPreprocess())
    model = model.to(device)
    conv_idx = 0

    for layer in pretrained_vgg.vgg.features.children():
        
        name = ""
        valid_layer = False
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f"conv_{conv_idx}"
            valid_layer = True
        elif isinstance(layer, nn.ReLU):
            layer.inplace = False
            name = f"relu_{conv_idx}"
            valid_layer = True
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_idx}"
            valid_layer = True
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_idx}"
            valid_layer = True
        
        if valid_layer:
            model.add_module(name, layer)

            if name in content_layers:
                # Content Loss
                target = model(content_spec).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{conv_idx}", content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # Style Loss
                target_feature = model(style_spec).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{conv_idx}", style_loss)
                style_losses.append(style_loss)
    
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def neural_style_transfer(model: nn.Module, 
                          content_spec: np.ndarray, 
                          style_spec: np.ndarray, 
                          device: torch.device = torch.device("cpu"), 
                          steps = 750, 
                          content_weight = 1, 
                          style_weight = 1000000): # 1000 (How much of style speaker to impose on content)

    cs_tensor = torch.Tensor(content_spec).float().to(device)
    ss_tensor = torch.Tensor(style_spec).float().to(device)

    model = model.to(device)

    if isinstance(model, VGG19Net):
        content_layers = ["conv_4"]
        style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]  # "conv_1", "conv_2", "conv_3", "conv_4", "conv_5"

        transfer_model, style_losses, content_losses = build_vgg_transfer_model(model, cs_tensor, ss_tensor, content_layers, style_layers, device)
    else:
        raise RuntimeError("Invalid CNN used in style transfer.")

    # Stylized output
    os_tensor = torch.Tensor(content_spec.copy())

    os_tensor = os_tensor.to(device)
    os_tensor.requires_grad_(True)

    transfer_model.eval()
    transfer_model.requires_grad_(False)

    optimizer = torch.optim.LBFGS([os_tensor])

    idx = [0]

    while idx[0] < steps:

        def closure():

            optimizer.zero_grad()
            transfer_model(os_tensor)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            idx[0] += 1
            if idx[0] % 50 == 0:
                bar = progbar(idx[0], steps)
                msg = "{} | Run {}: Style Loss : {:4f} | Content Loss: {:4f}".format(bar, idx, style_score.item(), content_score.item())
                stream(msg)


            return style_score + content_score

        optimizer.step(closure)
        
    print("\n")
    return os_tensor.cpu().detach().numpy()




    
    
    





