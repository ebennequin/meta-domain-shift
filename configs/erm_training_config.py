from functools import partial
from torch.optim import Adam

BATCH_SIZE = 128
N_EPOCHS = 100
N_TRAINING_IMAGES_PER_EPOCH = 100000
N_VAL_IMAGES_PER_EPOCH = 1000
N_WORKERS = 28
OPTIMIZER = partial(
    Adam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False
)
