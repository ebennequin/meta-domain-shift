import json
from pathlib import Path

import torch
from PIL import Image
from typing import Callable, Optional

import numpy as np
from torchvision.datasets import VisionDataset
from torchvision import transforms

from configs.dataset_specs.tiered_imagenet_c.perturbation_params import (
    PERTURBATION_PARAMS,
)
from src.datasets.transform import TransformLoader
from src.datasets.utils import get_perturbations


class TieredImageNetC(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        target_transform: Optional[Callable] = None,
    ):
        transform = TransformLoader(image_size).get_composed_transform(aug=False)
        super(TieredImageNetC, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        # We need to write this import here (and not at the top) to avoid cyclic imports
        from configs.dataset_config import SPECS_ROOT

        with open(SPECS_ROOT / f"{split}.json", "r") as file:
            split_specs = json.load(file)

        self.root = Path(root)
        self.class_list = split_specs["class_names"]
        self.id_to_class = dict(enumerate(self.class_list))
        self.images, self.labels = self.get_images_and_labels()

        self.perturbations, self.id_to_domain = get_perturbations(
            split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )

    def __len__(self):
        return len(self.labels) * len(self.perturbations)

    def __getitem__(self, item):
        original_data_index = item // len(self.perturbations)
        perturbation_index = item % len(self.perturbations)

        label = self.labels[original_data_index]

        img = Image.open(self.images[original_data_index]).convert("RGB")
        img = transforms.Resize((224, 224))(img)
        img = self.perturbations[perturbation_index](img)
        if isinstance(img, np.ndarray):
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.float32)

        assert img.dtype == torch.float32, self.perturbations[perturbation_index]
        assert img.shape == torch.Size([3, 224, 224])

        if self.target_transform is not None:
            label = self.target_transform(label)

        assert img.max() < 1.5, self.id_to_domain[int(perturbation_index)]

        return img, label, perturbation_index

    def get_images_and_labels(self):
        """
        Provides image paths and corresponding labels, as expected to define our VisionDataset objects.
        Returns:
            tuple(list(str), list(int): respectively the list of all paths to images belonging in the split defined in
            the input JSON file, and their class ids
        """

        image_names = []
        image_labels = []

        for class_id, class_name in enumerate(self.class_list):
            class_images_paths = [
                str(image_path)
                for image_path in (self.root / class_name).glob("*")
                if image_path.is_file()
            ]
            image_names += class_images_paths
            image_labels += len(class_images_paths) * [class_id]

        return image_names, image_labels
