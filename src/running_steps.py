from distutils.dir_util import copy_tree
from pathlib import Path
import random
from shutil import rmtree

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from configs import (
    dataset_config,
    training_config,
    model_config,
    experiment_config,
    evaluation_config,
)
from src.utils import set_device, elucidate_ids, get_episodic_loader


def prepare_output():
    if experiment_config.SAVE_RESULTS:
        if experiment_config.OVERWRITE & experiment_config.SAVE_DIR.exists():
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


def set_and_print_random_seed():
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Returns:
        int: numpy random seed

    """
    random_seed = experiment_config.RANDOM_SEED
    if not random_seed:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)
    torch.manual_seed(np.random.randint(0, 2 ** 32 - 1))
    random.seed(np.random.randint(0, 2 ** 32 - 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prompt = f"Random seed : {random_seed}"
    logger.info(prompt)

    return random_seed


def train_model():
    logger.info(
        "Initializing data loaders for {dataset}...",
        dataset=dataset_config.DATASET.__name__,
    )
    train_loader, _ = get_episodic_loader(
        "train",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_EPISODES,
    )
    val_loader, _ = get_episodic_loader(
        "val",
        n_way=training_config.N_WAY,
        n_source=training_config.N_SOURCE,
        n_target=training_config.N_TARGET,
        n_episodes=training_config.N_VAL_TASKS,
    )
    if training_config.TEST_SET_VALIDATION_FREQUENCY:
        test_loader, _ = get_episodic_loader(
            "test",
            n_way=training_config.N_WAY,
            n_source=training_config.N_SOURCE,
            n_target=training_config.N_TARGET,
            n_episodes=training_config.N_VAL_TASKS,
        )

    logger.info("Initializing model...")

    model = set_device(model_config.MODEL(model_config.BACKBONE))
    optimizer = training_config.OPTIMIZER(model.parameters())

    max_acc = -1.0
    best_model_epoch = -1
    best_model_state = None

    writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)

    logger.info("Model and data are ready. Starting training...")
    for epoch in range(training_config.N_EPOCHS):
        # Set model to training mode
        model.train()
        # Execute a training loop of the model
        train_loss, train_acc = model.train_loop(epoch, train_loader, optimizer)
        writer.add_scalar("Train/loss", train_loss, epoch)
        writer.add_scalar("Train/acc", train_acc, epoch)
        # Set model to evaluation mode
        model.eval()
        # Evaluate on validation set
        val_loss, val_acc, _ = model.eval_loop(val_loader)
        writer.add_scalar("Val/loss", val_loss, epoch)
        writer.add_scalar("Val/acc", val_acc, epoch)

        # We make sure the best model is saved on disk, in case the training breaks
        if val_acc > max_acc:
            max_acc = val_acc
            best_model_epoch = epoch
            best_model_state = model.state_dict()
            torch.save(best_model_state, experiment_config.SAVE_DIR / "best_model.tar")

        if training_config.TEST_SET_VALIDATION_FREQUENCY:
            if (
                epoch % training_config.TEST_SET_VALIDATION_FREQUENCY
                == training_config.TEST_SET_VALIDATION_FREQUENCY - 1
            ):
                logger.info("Validating on test set...")
                _, test_acc, _ = model.eval_loop(test_loader)
                writer.add_scalar("Test/acc", test_acc, epoch)

    logger.info(f"Training over after {training_config.N_EPOCHS} epochs")
    logger.info("Retrieving model with best validation accuracy...")
    model.load_state_dict(best_model_state)
    logger.info(f"Retrieved model from epoch {best_model_epoch}")

    writer.close()

    return model


def load_model(state_path: Path):
    model = set_device(model_config.MODEL(model_config.BACKBONE))
    model.load_state_dict(torch.load(state_path))
    logger.info(f"Loaded model from {state_path}")

    return model


def eval_model(model):
    logger.info(
        "Initializing test data from {dataset}...",
        dataset=dataset_config.DATASET.__name__,
    )
    test_loader, test_dataset = get_episodic_loader(
        "test",
        n_way=evaluation_config.N_WAY,
        n_source=evaluation_config.N_SOURCE,
        n_target=evaluation_config.N_TARGET,
        n_episodes=evaluation_config.N_TASKS,
    )

    logger.info("Starting model evaluation...")
    model.eval()

    _, acc, stats_df = model.eval_loop(test_loader)

    stats_df = elucidate_ids(stats_df, test_dataset)

    stats_df.to_csv(experiment_config.SAVE_DIR / "evaluation_stats.csv", index=False)

    return acc
