from src.erm_training_steps import *
from src.running_steps import prepare_output, set_and_print_random_seed


def main():
    prepare_output()
    set_and_print_random_seed()

    train_loader, val_loader, n_classes = get_data()

    model = get_model(n_classes)

    best_model_state, best_model_epoch = train(model, train_loader, val_loader)

    wrap_up_training(best_model_state, best_model_epoch)


if __name__ == "__main__":
    main()
