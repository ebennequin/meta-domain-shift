# Reproducing results

## Run an experiment

- Run a complete experiment (training + evaluation with the same setting):
  
    ``` python -m scripts.run_experiment```

    - You will find the trained model in a `best_model.tar` file in the specified output directory for
    future use.
    
    - The final evaluation accuracy, along with the evolution of the training loss, is printed in the logs.
    The logs are saved in a `running.log` file in the output directory.

- Run a standard Empirical Risk Minimization training:

    ``` python -m scripts.erm_training```

    You will find the trained model in a `.tar` file in the specified output directory.

- Evaluate an already trained model:

    ``` 
    python -m scripts.eval_model \
    --model-path=/path/to/best_model.tar \
    --episodic=True --use-fc=False --force-ot=False
  ```
    - `episodic`: set to False if the model was trained with standard ERM (*i.e.* non episodic)
  
    - `use-fc`: use the last linear layer of the model (if it has one). Only useful for Transductive Fine-Tuning
    
    - `force-ot`: force the model to use Optimal Transport. Useful to evaluate a model both with and without OT in one run.

    - The final evaluation accuracy is printed in the logs.
    The logs are saved in a `running.log` file in the output directory.

## Configure an experiment

You can configure your experiments in `configs/`.

- Choose the dataset: select the appropriate import in `configs/dataset_config.py`
- Specify the dataset's config (location on disk, etc...): edit `configs/all_datasets_configs/<dataset>_config.py`
- Training config: edit `configs/training_config.py`. All parameters are the same for all experiments we ran, except `N_SOURCE` (1 or 5) and `N_TARGET` (8 or 16).
- Model config: edit `configs/model_config.py`. In our experiments:

    - Parameters of the `OptimalTransport` stay always the same;
    - `BATCHNORM` can be `ConventionalBatchNorm` or `TransductiveBatchNorm`;
    - `BACKBONE` is `Conv4` for CIFAR and FEMNIST, `ResNet18` for miniImageNet;
    - `MODEL` is where you choose the FSL method;
      - Set the `transportation` parameter to `None` in order to not use OT.
  ...