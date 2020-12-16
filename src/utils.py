import torch
from torch.utils.data import DataLoader

from configs import dataset_config
from src.datasets.samplers import MetaSampler
from src.datasets.utils import episodic_collate_fn


def set_device(x):
    """
    Switch a tensor to GPU if CUDA is available, to CPU otherwise
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device=device)


def get_loader(split: str, n_way: int, n_source: int, n_target: int, n_episodes: int):
    dataset = dataset_config.DATASET(
        dataset_config.DATA_ROOT, split, dataset_config.IMAGE_SIZE
    )
    sampler = MetaSampler(
        dataset,
        n_way=n_way,
        n_source=n_source,
        n_target=n_target,
        n_episodes=n_episodes,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=episodic_collate_fn,
    )
