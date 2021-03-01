"""
Steps used in scripts/erm_training.py
"""

from typing import OrderedDict

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, random_split, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import dataset_config, erm_training_config, experiment_config, model_config
from src.utils import set_device


def get_few_shot_split() -> (Dataset, Dataset):
    temp_train_set = dataset_config.DATASET(
        dataset_config.DATA_ROOT, "train", dataset_config.IMAGE_SIZE
    )
    temp_train_classes = len(temp_train_set.id_to_class)
    temp_val_set = dataset_config.DATASET(
        dataset_config.DATA_ROOT,
        "val",
        dataset_config.IMAGE_SIZE,
        target_transform=lambda label: label + temp_train_classes,
    )
    if hasattr(dataset_config.DATASET, "__name__"):
        if dataset_config.DATASET.__name__ == "CIFAR100CMeta":
            label_mapping = {
                v: k
                for k, v in enumerate(
                    list(temp_train_set.id_to_class.keys())
                    + list(temp_val_set.id_to_class.keys())
                )
            }
            temp_train_set.target_transform = (
                temp_val_set.target_transform
            ) = lambda label: label_mapping[label]

    return temp_train_set, temp_val_set


def get_non_few_shot_split(
    temp_train_set: Dataset, temp_val_set: Dataset
) -> (Subset, Subset):
    train_and_val_set = ConcatDataset(
        [
            temp_train_set,
            temp_val_set,
        ]
    )
    n_train_images = int(
        len(train_and_val_set) * erm_training_config.TRAIN_IMAGES_PROPORTION
    )
    return random_split(
        train_and_val_set,
        [n_train_images, len(train_and_val_set) - n_train_images],
        generator=torch.Generator().manual_seed(
            erm_training_config.TRAIN_VAL_SPLIT_RANDOM_SEED
        ),
    )


def get_data() -> (DataLoader, DataLoader, int):
    logger.info("Initializing data loaders...")

    temp_train_set, temp_val_set = get_few_shot_split()

    train_set, val_set = get_non_few_shot_split(temp_train_set, temp_val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=erm_training_config.BATCH_SIZE,
        num_workers=erm_training_config.N_WORKERS,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=erm_training_config.BATCH_SIZE,
        num_workers=erm_training_config.N_WORKERS,
    )
    # Assume that train and val classes are entirely disjoints
    n_classes = len(temp_val_set.id_to_class) + len(temp_train_set.id_to_class)

    return train_loader, val_loader, n_classes


def get_model(n_classes: int) -> nn.Module:
    logger.info(f"Initializing {model_config.BACKBONE.__name__}...")
    model = set_device(model_config.BACKBONE())
    model.trunk.add_module("fc", set_device(nn.Linear(model.final_feat_dim, n_classes)))
    model.loss_fn = nn.CrossEntropyLoss()
    model.optimizer = erm_training_config.OPTIMIZER(model.parameters())
    return model


def get_n_batches(data_loader: DataLoader, n_images_per_epoch: int) -> int:
    """
    Computes the number of batches in a training epoch from the intended number of seen images.
    """

    return min(n_images_per_epoch // erm_training_config.BATCH_SIZE, len(data_loader))


def fit(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> (nn.Module, float):
    model.optimizer.zero_grad()
    scores = model(images)
    loss = model.loss_fn(scores, labels)
    loss.backward()
    model.optimizer.step()

    return model, loss.item()


def training_epoch(
    model: nn.Module, data_loader: DataLoader, epoch: int, n_batches: int
) -> (nn.Module, float):
    loss_list = []
    model.train()

    with tqdm(
        zip(range(n_batches), data_loader),
        total=n_batches,
        desc=f"Epoch {epoch}",
    ) as tqdm_train:
        for batch_id, (images, labels, _) in tqdm_train:
            model, loss_value = fit(model, set_device(images), set_device(labels))

            loss_list.append(loss_value)

            tqdm_train.set_postfix(loss=np.asarray(loss_list).mean())

    return model, np.asarray(loss_list).mean()


def validation(model: nn.Module, data_loader: DataLoader, n_batches: int) -> float:
    val_acc_list = []
    model.eval()
    with tqdm(
        zip(range(n_batches), data_loader),
        total=n_batches,
        desc="Validation:",
    ) as tqdm_val:
        for _, (images, labels, _) in tqdm_val:
            val_acc_list.append(
                float(
                    (
                        model(set_device(images)).data.topk(1, 1, True, True)[1][:, 0]
                        == set_device(labels)
                    ).sum()
                )
                / len(labels)
            )
            tqdm_val.set_postfix(accuracy=np.asarray(val_acc_list).mean())

    return np.asarray(val_acc_list).mean()


def train(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
) -> (OrderedDict, int):
    writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
    n_training_batches = get_n_batches(
        train_loader, erm_training_config.N_TRAINING_IMAGES_PER_EPOCH
    )
    n_val_batches = get_n_batches(
        val_loader, erm_training_config.N_VAL_IMAGES_PER_EPOCH
    )
    max_val_acc = 0.0
    best_model_epoch = 0
    best_model_state = model.state_dict()
    logger.info("Model and data are ready. Starting training...")
    for epoch in range(erm_training_config.N_EPOCHS):

        model, average_loss = training_epoch(
            model, train_loader, epoch, n_training_batches
        )

        writer.add_scalar(
            "Train/loss",
            average_loss,
            epoch,
        )

        val_acc = validation(model, val_loader, n_val_batches)
        writer.add_scalar("Val/acc", val_acc, epoch)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_model_epoch = epoch
            best_model_state = model.state_dict()

    return best_model_state, best_model_epoch


def wrap_up_training(best_model_state: OrderedDict, best_model_epoch: int):
    logger.info(f"Training complete.")
    logger.info(f"Best model found after {best_model_epoch + 1} training epochs.")
    state_dict_path = (
        experiment_config.SAVE_DIR
        / f"{model_config.BACKBONE.__name__}_{dataset_config.DATASET.__name__ if hasattr(dataset_config.DATASET, '__name__') else dataset_config.DATASET.func.__name__}.tar"
    )
    torch.save(best_model_state, state_dict_path)
    logger.info(f"Model state dict saved in {state_dict_path}")
