from functools import partial

from torch.optim import Adam

# Parameters for the model training

N_WAY = 2
N_SOURCE = 3
N_TARGET = 3
N_EPISODES = 20
N_VAL_TASKS = 10
N_EPOCHS = 2
OPTIMIZER = partial(
    Adam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
)
