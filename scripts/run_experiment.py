import torch

from src.running_steps import (
    train_model,
    eval_model,
    set_and_print_random_seed,
    prepare_output,
)

"""
Run a complete experiment (training + evaluation)
"""

if __name__ == "__main__":

    prepare_output()

    set_and_print_random_seed()
    trained_model = train_model()
    torch.cuda.empty_cache()

    set_and_print_random_seed()
    eval_model(trained_model)
