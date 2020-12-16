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

## Run an experiment

Configure your experiment by changing the values in `configs/*.py`, then launch your experiment.
```python scripts/run_experiment.py```

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

