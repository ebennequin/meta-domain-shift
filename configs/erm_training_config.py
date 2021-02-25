from functools import partial
from torch.optim import Adam

BATCH_SIZE = 128
N_EPOCHS = 200
N_TRAINING_IMAGES_PER_EPOCH = 128000
N_VAL_IMAGES_PER_EPOCH = 12800
N_WORKERS = 6
OPTIMIZER = partial(
    Adam, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False
)
TRAIN_IMAGES_PROPORTION = 0.82
TRAIN_VAL_SPLIT_RANDOM_SEED = 1
