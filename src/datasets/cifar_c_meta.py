from functools import partial
import json
import os
from pathlib import Path
from PIL import Image
import pickle
from typing import Any, Callable, Optional

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

from src.datasets.perturbations import PERTURBATIONS
from src.datasets.transform import TransformLoader


class CIFAR100CMeta(CIFAR100):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 224,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):

        transform = TransformLoader(image_size).get_composed_transform(aug=False)

        super(CIFAR10, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        downloaded_list = self.train_list + self.test_list

        self._load_meta()

        # We need to write this import here (and not at the top) to avoid cyclic imports
        from configs.dataset_config import SPECS_ROOT
        with open(SPECS_ROOT / f"{split}.json", "r") as file:
            self.split_specs = json.load(file)

        split_class_idx = {
            self.class_to_idx[class_name]
            for class_name in self.split_specs["class_names"]
        }

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                items_to_keep = [
                    item
                    for item in range(len(entry["data"]))
                    if entry["fine_labels"][item] in split_class_idx
                ]
                self.data.append([entry["data"][item] for item in items_to_keep])
                self.targets.extend(
                    [entry["fine_labels"][item] for item in items_to_keep]
                )

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.perturbations = []
        for perturbation_name, severities in self.split_specs["perturbations"].items():
            for severity in severities:
                self.perturbations.append(
                    partial(PERTURBATIONS[perturbation_name], severity=severity)
                )

    def __len__(self):
        return len(self.data) * len(self.perturbations)

    def __getitem__(self, item):
        original_data_index = item // len(self.perturbations)
        perturbation_index = item % len(self.perturbations)

        img, target = (
            Image.fromarray(self.data[original_data_index]),
            self.targets[original_data_index],
        )

        img = self.perturbations[perturbation_index](img)

        if self.transform is not None:
            # TODO: some perturbations output arrays, some output images. We need to clean that.
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, perturbation_index
