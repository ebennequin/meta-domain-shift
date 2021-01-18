import random

import numpy as np
import torch
from torch.utils.data import Sampler


class MetaSampler(Sampler):
    def __init__(self, dataset, n_way, n_source, n_target, n_episodes):
        self.n_perturbations = len(dataset.perturbations)
        self.n_total_images = len(dataset.images)
        self.n_way = n_way
        self.n_source = n_source
        self.n_target = n_target
        self.n_episodes = n_episodes

        self.items_per_label = {}

        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_episodes

    def _get_episode_items(self):
        labels = random.sample(self.items_per_label.keys(), self.n_way)
        source_perturbation, target_perturbation = torch.randperm(self.n_perturbations)[
            :2
        ]

        source_items = (
            torch.cat(
                [
                    torch.tensor(
                        random.sample(self.items_per_label[label], self.n_source)
                    )
                    for label in labels
                ]
            )
            * self.n_perturbations
            + source_perturbation
        )
        target_items = (
            torch.cat(
                [
                    torch.tensor(
                        random.sample(self.items_per_label[label], self.n_target)
                    )
                    for label in labels
                ]
            )
            * self.n_perturbations
            + target_perturbation
        )

        return torch.cat((source_items, target_items))

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._get_episode_items()
