import collections
from functools import partial
import torch


def activation_filters(net):
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)

    def save_activation(name, out):
        activations[name].append(out.cpu())

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in net.named_modules():
        m.register_forward_hook(partial(save_activation, name))

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    return activations
