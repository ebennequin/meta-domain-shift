from pathlib import Path

import click
from loguru import logger

from src.running_steps import (
    load_model,
    eval_model,
    set_and_print_random_seed,
    prepare_output,
)

"""
Evaluate a trained model.
"""


@click.option(
    "--model-path",
    help="Path to the model state to be loaded",
    type=Path,
    required=True,
)
@click.option(
    "--episodic",
    help="Whether the model was trained using episodic training",
    type=bool,
    default=True,
)

@click.option(
    "--use_fc",
    help="Whether the model load the fc layer in the backbone",
    type=bool,
    default=False,
)

@click.command()
def main(model_path: Path, episodic: bool, use_fc: bool):
    prepare_output()
    set_and_print_random_seed()

    trained_model = load_model(model_path, episodic, use_fc)

    eval_model(trained_model)


if __name__ == "__main__":
    main()
