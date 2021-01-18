from functools import partial

from torch.optim import Adam

# Parameters for the model training

N_WAY = 5
N_SOURCE = 5
N_TARGET = 16
N_EPISODES = 400
N_VAL_TASKS = 1000
N_EPOCHS = 100
OPTIMIZER = partial(
    Adam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False
)
# Set the following to None to avoid using test set during training
TEST_SET_VALIDATION_FREQUENCY = 21
