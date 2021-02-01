from torch import nn
from torch.utils.data import DataLoader

from src.running_steps import *


def fit(model, images, labels):
    model.optimizer.zero_grad()
    loss = model.loss_fn(model(images), labels)
    loss.backward()
    model.optimizer.step()

    return model, loss.item()


def main():
    prepare_output()
    set_and_print_random_seed()

    logger.info(
        "Initializing data loaders for {dataset}...",
        dataset=dataset_config.DATASET.__name__,
    )
    train_set = dataset_config.DATASET(
        dataset_config.DATA_ROOT, "val", dataset_config.IMAGE_SIZE
    )
    train_loader = DataLoader(train_set, batch_size=64)
    n_classes = len(train_set.id_to_class)

    logger.info(f"Initializing {model_config.BACKBONE.__name__}...")
    model = set_device(model_config.BACKBONE())
    model.trunk.add_module("fc", set_device(nn.Linear(model.final_feat_dim, n_classes)))

    model.loss_fn = nn.CrossEntropyLoss()
    model.optimizer = training_config.OPTIMIZER(model.parameters())

    writer = SummaryWriter(log_dir=experiment_config.SAVE_DIR)
    mid_epoch_log_frequency = len(train_loader) // 10

    logger.info("Model and data are ready. Starting training...")
    for epoch in range(training_config.N_EPOCHS):

        model.train()
        loss_list = []

        for batch_id, (images, labels, _) in enumerate(train_loader):
            images = set_device(images)
            labels = set_device(labels)
            model, loss_value = fit(model, images, labels)

            loss_list.append(loss_value)

            if (batch_id + 1) % mid_epoch_log_frequency == 0:
                logger.info(
                    "Epoch {epoch} | Batch {episode_index}/{n_batches} | Loss {loss}".format(
                        epoch=epoch,
                        episode_index=batch_id + 1,
                        n_batches=len(train_loader),
                        loss=np.asarray(loss_list).mean(),
                    )
                )
        writer.add_scalar("Train/loss", np.asarray(loss_list).mean(), epoch)

        # TODO: add validation

    state_dict_path = (
        experiment_config.SAVE_DIR
        / f"{model_config.BACKBONE.__name__}_{dataset_config.DATASET.__name__}.tar"
    )
    torch.save(model.state_dict(), state_dict_path)
    logger.info(f"Training complete. Model state dict saved in {state_dict_path}")

    return model


if __name__ == "__main__":
    main()
