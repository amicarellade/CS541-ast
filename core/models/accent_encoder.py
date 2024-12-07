"""
Accent Encoder


Train Convnet like VGG on identifying spectrograms / waveforms that pertain to certain accents
From there use pretrained frozen network to perform Gatys style transfer

"""

import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


""" ------ Models ------ """
    

class VGG19Net(nn.Module):
    """
    VGG19 Wrapper Class (Used for running predictions against mel spectrograms)
    """

    def __init__(self, num_classes: int):
        super(VGG19Net, self).__init__()

        self.vgg = torchvision.models.vgg19()
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)


""" ------ Loss Functions and other computations ------ """


class ContentLoss(nn.Module):

    def __init__(self, content: torch.Tensor):
        super(ContentLoss, self).__init__()

        self.content = content.detach()


    def forward(self, x):
        self.loss = F.mse_loss(x, self.content)
        return x


class StyleLoss(nn.Module):

    def __init__(self, style: torch.Tensor):
        super(StyleLoss, self).__init__()

        self.style = gram_matrix(style).detach()


    def forward(self, x):
        self.loss = F.mse_loss(gram_matrix(x), self.style)
        return x


class VGGPreprocess(nn.Module):

    def __init__(self):
        super(VGGPreprocess, self).__init__()

        self.requires_grad_(False)

        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to VGG19 input size
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])


    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(0).repeat_interleave(3, dim=0)
        return self.transform(x)


def gram_matrix(in_tensor: torch.Tensor):
    """
    Normalized Gram Matrix
    """

    if len(in_tensor.size()) == 3:
        in_tensor = in_tensor.unsqueeze(0)

    a, b, c, d = in_tensor.size()

    features = in_tensor.view(a * b, c * d)

    G = torch.mm(features, features.T)

    return G.div(a * b * c * d)





