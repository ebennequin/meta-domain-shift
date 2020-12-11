from loguru import logger
import numpy as np
import torch


def set_device(x):
    """
    Switch a tensor to GPU if CUDA is available, to CPU otherwise
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device=device)


def set_and_print_random_seed(random_seed=None, save_dir=None):
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Args:
        random_seed (int): seed for random instantiations ; if none is provided, a seed is randomly defined
        save_dir (Path): output folder where the seed is saved
    Returns:
        int: numpy random seed

    """
    if not random_seed:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)
    torch.manual_seed(np.random.randint(0, 2 ** 32 - 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    prompt = f"Random seed : {random_seed}"
    logger.info(prompt)

    if save_dir:
        with open(save_dir / "seeds.txt", "a") as f:
            f.write(prompt)

    return random_seed
