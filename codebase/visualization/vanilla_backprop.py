"""
This code is modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations
Original author: Utku Ozbulak - github.com/utkuozbulak
"""

import sys

sys.path.append("..")

import torch


class VanillaBackprop:
    """
        Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model):
        self.model = model

    def generate_gradients(self, input_image, target_class, loss):
        # Put model in evaluation mode
        self.model.eval()

        x = input_image.clone()

        x.requires_grad = True

        outputs = self.model(x)
        self.model.zero_grad()
        assert x.grad is None, "x should not have gradients yet."

        cost = loss(outputs, target_class)
        cost.backward()

        grad = x.grad.detach()

        return grad
