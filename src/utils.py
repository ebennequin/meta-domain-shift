import re

from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision


def set_device(x):
    """
    Switch a tensor to GPU if CUDA is available, to CPU otherwise
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device=device)


def plot_episode(support_images, query_images):
    """
    Plot images of an episode, separating support and query images.
    Args:
        support_images (torch.Tensor): tensor of multiple-channel support images
        query_images (torch.Tensor): tensor of multiple-channel query images
    """

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    support_grid = torchvision.utils.make_grid(support_images)
    matplotlib_imshow(support_grid)
    plt.title("support images")
    plt.show()
    query_grid = torchvision.utils.make_grid(query_images)
    plt.title("query images")
    matplotlib_imshow(query_grid)
    plt.show()


def clean_name(name, banned_characters="[^A-Za-z0-9]+", fill_item="_"):
    return re.sub(banned_characters, fill_item, name)
