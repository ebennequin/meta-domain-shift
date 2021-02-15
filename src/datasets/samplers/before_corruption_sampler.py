import random

import torch
from torch.utils.data import Sampler

from sklearn.model_selection import train_test_split


class BeforeCorruptionSampler(Sampler):
    """
    Sample images from a dataset which uses image perturbations (like CIFAR100-C or tieredImageNet-C), in the
    case where perturbations are applied online. For the case where images on disk are already corrupted, see
    AfterCorruptionSampler.
    """

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

        self.source_items_per_label = {}
        self.target_items_per_label = {}

    def __len__(self):
        return self.n_episodes

    def _split_source_target(self):
        self.source_items_per_label = {}
        self.target_items_per_label = {}

        for label in list(self.items_per_label.keys()):
            self.source_items_per_label[label], self.target_items_per_label[label] = train_test_split(
                self.items_per_label[label], 
                train_size=0.5
            )

    def _sample_instances_from_label(self, label, n_samples, mode):
        if mode == 'source':
            items_per_label = dict(self.source_items_per_label)
        elif mode == 'target':
            items_per_label = dict(self.target_items_per_label)
        
        return torch.tensor(
            items_per_label[label]
            if n_samples == -1
            else random.sample(items_per_label[label], n_samples)
        )

    def _get_episode_items(self):
        self._split_source_target()

        labels = random.sample(self.items_per_label.keys(), self.n_way)
        source_perturbation, target_perturbation = torch.randperm(self.n_domains)[:2]

        source_items = (
            torch.cat(
                [
                    self._sample_instances_from_label(label, self.n_source, 'source')
                    for label in labels
                ]
            )
            * self.n_domains
            + source_perturbation
        )

        target_items = (
            torch.cat(
                [
                    self._sample_instances_from_label(label, self.n_target, 'target')
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
