import torch
from polyaxon_client.tracking import Experiment

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
    # Warning: this fails if outside a polyaxon managed container
    experiment = Experiment()

    prepare_output()

    set_and_print_random_seed()
    trained_model = train_model()
    torch.cuda.empty_cache()

    set_and_print_random_seed()
    acc = eval_model(trained_model)

    experiment.log_metrics(accuracy=acc)
