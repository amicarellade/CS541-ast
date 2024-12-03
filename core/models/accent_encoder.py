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


""" ------ Models ------ """
    

class VGG19Net(nn.Module):
    """
    VGG19 Wrapper Class (Used for running predictions against mel spectrograms)
    """

    def __init__(self):
        super(VGG19Net, self).__init__()
        self.vgg = torchvision.models.vgg19_bn()

    def forward(self, x):
        return self.vgg(x)

    def predict(self, x, n_classes: int = 11):
        return F.softmax(self.forward(x), n_classes)


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


def gram_matrix(in_tensor: torch.Tensor):
    """
    Normalized Gram Matrix
    """

    a, b, c, d = in_tensor.size()

    features = in_tensor.view(a * b, c * d)

    G = torch.mm(features, features.T())

    return G.div(a * b * c * d)





