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

Then `cd meta-domain-shift` and install virtualenv:

```
virtualenv venv --python=python3.7
source venv/bin/activate
```

Then install dependencies: `pip install -r requirements.txt`.

## Data
To install the datasets to your machine, please follow [this walkthrough](DATASETS.md).

## Run an experiment

Configure your experiment by changing the values in `configs/*.py`, then launch your experiment.
```python scripts/run_experiment.py```

All outputs of the experiment (explicit configuration, logs, trained model state and TensorBoard logs) 
can then be found in the directory specified in `configs/experiment_config.py`.

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

