from functools import partial

from torch.optim import Adam

# Parameters for the model training

N_WAY = 10
N_SOURCE = 20
N_TARGET = 64
N_EPISODES = 400
N_VAL_TASKS = 1000
N_EPOCHS = 100
OPTIMIZER = partial(
    Adam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False
)
