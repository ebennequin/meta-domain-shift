# Meta Domain Shift

## Enviroment
 - Python 3.7
 - [Pytorch](http://pytorch.org/) 1.7
 - CUDA 10

## Getting started

Git clone the repo:

```
git clone git@github.com:ebennequin/meta-domain-shift.git
```

Then `cd meta-domain-shift` and create a virtual environment (if you don't already have it,
use `pip install virtualenv`):

```
virtualenv venv --python=python3.7
source venv/bin/activate
```

Then install dependencies: `pip install -r requirements.txt`
Some perturbations used in CIFAR-100-C-FewShot and *mini*ImageNet-C use Wand: `sudo apt-get install libmagickwand-dev`

## Data
To install the datasets to your machine, please follow [this walkthrough](DATASETS.md).

## Run an experiment

Configure your experiment by changing the values in `configs/*.py`, then launch your experiment.
```python -m scripts.run_experiment```

On some machines, the `src` module will not be found by Python. If this happens to you, run
`export PYTHONPATH=$PYTHONPATH:path/to/meta-domain-shift` to tell Python where you're at.

All outputs of the experiment (explicit configuration, logs, trained model state and TensorBoard logs) 
can then be found in the directory specified in `configs/experiment_config.py`. By default, an error will be risen if 
the specified directory already exists (in order to not harm the results of previous experiments). You may
change this behaviour in `configs/experiment_config.py` by setting `OVERWRITE = True`.

### Reproducing results

See the detailed documentation [here](REPRODUCING.md).

### Track trainings with Tensorboard

We log the loss and validation accuracy during the training for visualization in Tensorboard. The logs of an
experiment can be found in the output directory (`events.out.tfevents.[...]`). To visualize them in Tensorboard, run:
```
tensorboard --logdir=output_dir
```

## Contribute

1. Make sure you start from an up to date `master` branch
   ```
   git checkout master && git pull
   ```
2. Create a new branch
    ```
    git checkout -b my-branch
   ```
3. Add your changes
    ```
   git add -p
     ```
4. Commit your changes
    ``` 
   git commit -m "Add FEMNIST dataset"
    ```
5. Push your changes to the `origin` remote
    ``` 
   git push --set-upstream origin my-branch
    ```
6. Open a pull request on GitHub and ask a teammate to review your code
7. Once reviewed, merge your branch to master

### Using a new library

If your changes to the code need a new library (not in the requirements), please add it to `requirements.txt`


## References
Meta-learning code is modified from https://github.com/sicara/FewShotLearning
Image perturbations are modified from https://github.com/hendrycks/robustness

