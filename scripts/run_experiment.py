from distutils.dir_util import copy_tree
from shutil import rmtree

from loguru import logger
import torch
from torch.utils.data import DataLoader

from configs import (
    dataset_config,
    evaluation_config,
    model_config,
    training_config,
    experiment_config,
)
from src.datasets.samplers import MetaSampler
from src.datasets.utils import episodic_collate_fn
from src.utils import set_and_print_random_seed, set_device

"""
Run a complete experiment (training + evaluation)
"""


def train_model():
    logger.info(
        "Initializing data loaders for {dataset}...",
        dataset=dataset_config.DATASET.__name__,
    )
    train_loader = get_loader(
        "train",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_EPISODES,
    )
    val_loader = get_loader(
        "val",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_VAL_TASKS,
    )

    try:
        logger.info(
            "Initializing {model} with {backbone}...",
            model=model_config.MODEL.__name__,
            backbone=model_config.BACKBONE.__name__,
        )
    except AttributeError:
        logger.info("Initializing model...")

    model = set_device(model_config.MODEL(model_config.BACKBONE))
    optimizer = training_config.OPTIMIZER(model.parameters())

    max_acc = -1.0
    best_model_epoch = -1
    best_model_state = None

    logger.info("Model and data are ready. Starting training...")
    for epoch in range(training_config.N_EPOCHS):
        # Set model to training mode
        model.train()
        # Execute a training loop of the model
        model.train_loop(epoch, train_loader, optimizer)
        # Set model to evaluation mode
        model.eval()
        # Evaluate on validation set
        _, acc, _ = model.eval_loop(val_loader)

        # We make sure the best model is saved on disk, in case the training breaks
        if acc > max_acc:
            max_acc = acc
            best_model_epoch = epoch
            best_model_state = model.state_dict()
            torch.save(best_model_state, experiment_config.SAVE_DIR / "best_model.tar")

    logger.info(f"Training over after {training_config.N_EPOCHS} epochs")
    logger.info("Retrieving model with best validation accuracy...")
    model.load_state_dict(best_model_state)
    logger.info(f"Retrieved model from epoch {best_model_epoch}")

    return model


def eval_model(model):
    logger.info(
        "Initializing test data from {dataset}...",
        dataset=dataset_config.DATASET.__name__,
    )
    test_loader = get_loader(
        "test",
        n_way=evaluation_config.N_WAY,
        n_source=evaluation_config.N_SOURCE,
        n_target=evaluation_config.N_TARGET,
        n_episodes=evaluation_config.N_TASKS,
    )

    logger.info("Starting model evaluation...")
    model.eval()

    _, acc, _ = model.eval_loop(test_loader)

    return acc


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


def prepare_output():
    if experiment_config.SAVE_RESULTS:
        if experiment_config.OVERWRITE:
            rmtree(str(experiment_config.SAVE_DIR))
            logger.info(
                "Deleting previous content of {directory}",
                directory=experiment_config.SAVE_DIR,
            )

        experiment_config.SAVE_DIR.mkdir(parents=True, exist_ok=False)
        logger.add(experiment_config.SAVE_DIR / "running.log")
        copy_tree("configs", str(experiment_config.SAVE_DIR / "experiment_parameters"))
        logger.info(
            "Parameters and outputs of this experiment will be saved in {directory}",
            directory=experiment_config.SAVE_DIR,
        )

    else:
        logger.info("This experiment will not be saved on disk.")


if __name__ == "__main__":

    prepare_output()
    set_and_print_random_seed()

    trained_model = train_model()

    eval_model(trained_model)
