import random

import numpy as np
import torch
from torch.utils.data import Sampler


class MetaSampler(Sampler):
    def __init__(self, dataset, n_way, n_source, n_target, n_episodes):
        self.n_domains = len(dataset.id_to_domain)
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
        source_perturbation, target_perturbation = torch.randperm(self.n_domains)[:2]

        def get_rand_item(item, n):
            if n == -1: 
                return item
            else: 
                return random.sample(item, n)

        source_items = (
            torch.cat(
                [
                    torch.tensor(
                        self.items_per_label[label] if self.n_source == -1 \
                            else random.sample(self.items_per_label[label], self.n_source)
                    )
                    for label in labels
                ]
            )
            * self.n_domains
            + source_perturbation
        )

        target_items = (
            torch.cat(
                [
                    torch.tensor(
                        self.items_per_label[label] if self.n_target == -1 \
                            else random.sample(self.items_per_label[label], self.n_target)
                    )
                    for label in labels
                ]
            )
            * self.n_domains
            + target_perturbation
        )

        return torch.cat((source_items, target_items))

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._get_episode_items()
