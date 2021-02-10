# Install datasets

We provide three new benchmarks for the problem of Few-Shot Image Classification with Support/Query Shift:

- CIFAR100-C-FewShot
- *tiered*ImageNet-C
- FEMNIST-FewShot

These are the instructions ot get the datasets on your machine.

## CIFAR100-C-FewShot

CIFAR1OO-C-FewShot is built from CIFAR100, which is easily downloadable with PyTorch. To download it, simply run the 
following python commands once:
```python
from torchvision.datasets import CIFAR100
CIFAR100("./data", download=True)
```
This will download CIFAR100 in `./data/cifar-100-python`. The perturbations used to make CIFAR100-C
are applied to the images online during runtime.

## *tiered*ImageNet-C

This dataset is built on the train split of ILSVRC2015 (available for download [here](http://www.image-net.org/)).
To run the experiments, you will only need to specify in `./configs/dataset_config.py` as DATA_ROOT the directory
where the ImageNet classes are located.

As CIFAR100-C, this dataset is built using [Hendrycks' perturbations](https://github.com/hendrycks/robustness).
By default, these perturbations are applied online to the images when loaded from the disk during runtime.
This is very costly in CPU usage. 

If this doesn't work on your machine, run 
`python -m scripts/datasets/write_tieredimagenet_c.py --input-dir=path/to/ILSVRC2015/Data/CLS-LOC/train --output-dir=where/you/want` 
to write the corrupted dataset explicitly on your disk. This process may take a long time to run and will need TB of disk
space, but it will bless you with smooth and quick experiments on *tiered*ImageNet-C. To use the corrupted dataset in your
experiments, in `./configs/dataset_config.py`, define `DATASET = partial(TieredImageNetC, load_corrupted_dataset=True)`.



## FEMNIST-FewShot
Federated-EMNIST is a version of the EMNIST dataset, where images are sorted by writer (or user).
FEMNIST-FewShot consists in a split of the FEMNIST dataset adapted to few-shot classification: 
we separate both users and classes between train, val and test sets.

First, you need to download the FEMNIST dataset into `./data/femnist` 
by running `source scripts/datasets/download_femnist.sh`
Then run `python -m scripts.datasets.preprocess_femnist.py`. This will write the images of FEMNIST-FewShot
(defined in the specification files `./configs/dataset_specs/femnist/*.csv`) into three numpy files, for train,
val and test sets. 

The code in this repository only uses the `csv` and `.npy` files, so once the previous steps are
done, feel free to delete `./data/femnist/by_write.zip` and `./data/femnist/by_write/`.

Check out [this repo](https://github.com/TalwalkarLab/leaf/) to explore the original FEMNIST dataset in more details.