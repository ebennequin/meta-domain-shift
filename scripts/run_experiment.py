from loguru import logger
from torch.utils.data import DataLoader

from configs import *
from src.datasets.samplers import MetaSampler
from src.datasets.utils import episodic_collate_fn
from src.utils import set_and_print_random_seed, set_device

'''
Run a complete experiment (training + evaluation)
'''

def train_model():
    logger.info("Initializing data loaders...")
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

    logger.info("Initializing model...")
    model = set_device(model_config.MODEL(model_config.BACKBONE))
    optimizer = training_config.OPTIMIZER(model.parameters())

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

    logger.info(f"Training over after {training_config.N_EPOCHS} epochs")
    return model


def eval_model(model):
    logger.info("Initializing test data...")
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
    dataset = dataset_config.DATASET("data", split)
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


if __name__ == "__main__":

    set_and_print_random_seed()

    trained_model = train_model()

    eval_model(trained_model)
