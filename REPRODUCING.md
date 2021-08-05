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

- To choose the dataset: select the appropriate import in `configs/dataset_config.py`
- To specify the dataset's config (location on disk, etc...): edit `configs/all_datasets_configs/<dataset>_config.py`
- Training config: edit `configs/training_config.py`. All parameters are the same for all experiments we ran, except `N_SOURCE` (1 or 5) and `N_TARGET` (8 or 16).
- Model config: edit `configs/model_config.py`. In our experiments:

    - Parameters of the `OptimalTransport` stay always the same;
    - `BATCHNORM` can be `ConventionalBatchNorm` or `TransductiveBatchNorm`;
    - `BACKBONE` is `Conv4` for CIFAR and FEMNIST, `ResNet18` for miniImageNet;
    - `MODEL` is where you choose the FSL method:
      - Select the method (`MatchingNet`, `ProtoNet`, `TransPropNet`, `TransFineTune`)
      - Set the `transportation` parameter to `None` in order to not use OT.
  
- Evaluation config: edit `configs/evaluation_config.py` to change the size and number of evaluation tasks.
Set `SUPPORT_QUERY_SHIFT` to False in order to sample tasks where source and target domains are the same.
  
- Training config specifically for standard ERM training: edit `configs/erm_training_config.py`. 
  We adapt the number of epochs and number of images per epoch to ensure that two models respectively trained
  with episodic training and standard ERM training roughly see the same number of images.
  
- Finally, edit `configs/experiment_config.py` to change general settings of your experiment (mostly the random seed and output directory).

## Results

Here are the extended results as shown in the paper. 
You are very welcome to improve this table with your own results with new methods / datasets!

**Meta-CIFAR100-C**


|                                      | 1-shot 8-target | 1-shot 16-target | 5-shot 8-target | 1-shot 16-target |
|--------------------------------------|-----------------|------------------|-----------------|------------------|
| ProtoNet + ET + CBN                  | 30.02           | 29.98            | 42.77           | 42.07            |
| ProtoNet + ET + CBN w. OT-TT         | 32.11           | 32.24            | 47.54           | 48.26            |
| ProtoNet + ET + CBN w. OT            | 33.74           | 35.63            | 48.37           | 48.25            |
| ProtoNet + ET + TBN                  | 32.47           | 32.52            | 48.00           | 46.49            |
| ProtoNet + ET + TBN w. OT-TT         | 32.81           | 31.72            | 48.62           | 48.71            |
| ProtoNet + ET + TBN w. OT            | 34.00           | **36.20**            | **49.71**          | **49.94**            |
| ProtoNet + ERM + CBN                 | 29.10           | 29.02            | 44.89           | 44.67            |
| ProtoNet + ERM + CBN w. OT           | 35.48           | 35.89            | 48.61           | 48.61            |
| ProtoNet + ERM + TBN                 | 29.49           | 29.61            | 46.59           | 46.48            |
| ProtoNet + ERM + TBN w. OT           | 35.40           | 35.94            | 48.66           | 48.89            |
| MatchingNet + ET + CBN               | 30.71           | 31.10            | 41.15           | 41.74            |
| MatchingNet + ET + CBN w. OT-TT      | 32.85           | 30.94            | 43.90           | 44.51            |
| MatchingNet + ET + CBN w. OT         | 34.48           | 35.53            | 44.55           | 45.71            |
| MatchingNet + ET + TBN               | 32.97           | 33.08            | 45.05           | 44.91            |
| MatchingNet + ET + TBN w. OT-TT      | 32.78           | 33.28            | 44.86           | 44.71            |
| MatchingNet + ET + TBN w. OT         | 35.11           | **36.36**            | 45.78           | 47.37            |
| MatchingNet + ERM + CBN              | 33.50           | 33.49            | 43.00           | 42.97            |
| MatchingNet + ERM + CBN w. OT        | **36.13**           | **36.61**            | 45.35           | 46.06            |
| MatchingNet + ERM + TBN              | 33.67           | 33.64            | 43.51           | 46.22            |
| MatchingNet + ERM + TBN w. OT        | **35.87**           | **36.54**            | 45.10           | 46.37            |
| TransPropNet + ET + CBN              | 30.26           | 30.82            | 39.13           | 38.73            |
| TransPropNet + ET + CBN w. OT-TT     | 28.70           | 32.39            | 40.60           | 39.25            |
| TransPropNet + ET + CBN w. OT        | 26.87           | 31.15            | 25.68           | 37.22            |
| TransPropNet + ET + TBN              | 34.15           | 34.83            | 47.39           | 43.91            |
| TransPropNet + ET + TBN w. OT-TT     | 29.48           | 33.53            | 40.47           | 40.62            |
| TransPropNet + ET + TBN w. OT        | 27.68           | 31.33            | 27.29           | 40.02            |
| TransPropNet + ERM + CBN             | 23.33           | 26.81            | 29.32           | 33.06            |
| TransPropNet + ERM + CBN w. OT       | 31.08           | 33.90            | 39.82           | 40.03            |
| TransPropNet + ERM + TBN             | 22.55           | 27.9             | 29.50           | 33.93            |
| TransPropNet + ERM + TBN w. OT       | 31.20           | 31.10            | 29.82           | 40.03            |
| Transductive Fine-Tuning + ERM + CBN | 28.91           | 29.01            | 37.28           | 37.51            |
| Transductive Fine-Tuning + ERM + TBN | 28.75           | 28.86            | 37.40           | 37.66            |     |                  |                 |                  |

**miniImageNet-C**

|                                      | 1-shot 8-target | 1-shot 16-target | 5-shot 8-target | 1-shot 16-target |
|--------------------------------------|-----------------|------------------|-----------------|------------------|
| ProtoNet + ET + CBN                  | 36.37           |                  | 47.58           |                  |
| ProtoNet + ET + CBN w. OT-TT         |                 |                  |                 |                  |
| ProtoNet + ET + CBN w. OT            |                 |                  |                 |                  |
| ProtoNet + ET + TBN                  | 40.43           |                  | 53.71           |                  |
| ProtoNet + ET + TBN w. OT-TT         | **44.77**           |                  | **60.46**          |                  |
| ProtoNet + ET + TBN w. OT            | 40.49           |                  | 59.85           |                  |
| ProtoNet + ERM + CBN                 |                 |                  |                 |                  |
| ProtoNet + ERM + CBN w. OT           |                 |                  |                 |                  |
| ProtoNet + ERM + TBN                 |                 |                  |                 |                  |
| ProtoNet + ERM + TBN w. OT           | 42.46           |                  | 54.67           |                  |
| MatchingNet + ET + CBN               | 35.26           |                  | 44.75           |                  |
| MatchingNet + ET + CBN w. OT-TT      |                 |                  |                 |                  |
| MatchingNet + ET + CBN w. OT         |                 |                  |                 |                  |
| MatchingNet + ET + TBN               |                 |                  |                 |                  |
| MatchingNet + ET + TBN w. OT-TT      |                 |                  |                 |                  |
| MatchingNet + ET + TBN w. OT         |                 |                  |                 |                  |
| MatchingNet + ERM + CBN              |                 |                  |                 |                  |
| MatchingNet + ERM + CBN w. OT        |                 |                  |                 |                  |
| MatchingNet + ERM + TBN              |                 |                  |                 |                  |
| MatchingNet + ERM + TBN w. OT        |                 |                  |                 |                  |
| TransPropNet + ET + CBN              |                 |                  |                 |                  |
| TransPropNet + ET + CBN w. OT-TT     |                 |                  |                 |                  |
| TransPropNet + ET + CBN w. OT        |                 |                  |                 |                  |
| TransPropNet + ET + TBN              | 24.10           |                  | 27.24           |                  |
| TransPropNet + ET + TBN w. OT-TT     |                 |                  |                 |                  |
| TransPropNet + ET + TBN w. OT        |                 |                  |                 |                  |
| TransPropNet + ERM + CBN             |                 |                  |                 |                  |
| TransPropNet + ERM + CBN w. OT       |                 |                  |                 |                  |
| TransPropNet + ERM + TBN             |                 |                  |                 |                  |
| TransPropNet + ERM + TBN w. OT       |                 |                  |                 |                  |
| Transductive Fine-Tuning + ERM + CBN |                 |                  |                 |                  |
| Transductive Fine-Tuning + ERM + TBN | 39.02           |                  | 51.27           |                  |

**FEMNIST-FewShot**

|                                      | 1-shot 1-target |
|--------------------------------------|-----------------|
| ProtoNet + ET + CBN                  | 84.31           |
| ProtoNet + ET + CBN w. OT-TT         | 94.00           |
| ProtoNet + ET + CBN w. OT            | 92.31           |
| ProtoNet + ET + TBN                  | 90.36           |
| **ProtoNet + ET + TBN w. OT-TT  **       | **94.92**           |
| ProtoNet + ET + TBN w. OT            | 93.63           |
| ProtoNet + ERM + CBN                 | 80.20           |
| ProtoNet + ERM + CBN w. OT           | 94.30           |
| ProtoNet + ERM + TBN                 | 86.22           |
| ProtoNet + ERM + TBN w. OT           | 94.22           |
| MatchingNet + ET + CBN               | 84.25           |
| MatchingNet + ET + CBN w. OT-TT      | 96.66           |
| MatchingNet + ET + CBN w. OT         | 92.73           |
| MatchingNet + ET + TBN               | 91.05           |
| **MatchingNet + ET + TBN w. OT-TT**      | **95.37**           |
| MatchingNet + ET + TBN w. OT         | 93.62           |
| MatchingNet + ERM + CBN              | 85.04           |
| MatchingNet + ERM + CBN w. OT        | 94.34           |
| MatchingNet + ERM + TBN              | 87.19           |
| MatchingNet + ERM + TBN w. OT        | 94.26           |
| TransPropNet + ET + CBN              | 31.30           |
| TransPropNet + ET + CBN w. OT-TT     | 40.60           |
| TransPropNet + ET + CBN w. OT        | 79.30           |
| TransPropNet + ET + TBN              | 86.42           |
| TransPropNet + ET + TBN w. OT-TT     | 93.08           |
| TransPropNet + ET + TBN w. OT        | 87.52           |
| TransPropNet + ERM + CBN             | 45.36           |
| TransPropNet + ERM + CBN w. OT       | 73.64           |
| TransPropNet + ERM + TBN             | 47.34           |
| TransPropNet + ERM + TBN w. OT       | 79.50           |
| Transductive Fine-Tuning + ERM + CBN | 86.13           |
| Transductive Fine-Tuning + ERM + TBN | 85.92           |     |                  |                 |                  |

**Accuracy in absence of Support-Query Shift**

|                                  | Transported Prototypes |
|----------------------------------|------------------------|
| Meta-CIFAR100-C 1-shot 8-target  | 85.67                  |
| Meta-CIFAR100-C 1-shot 16-target | 88.52                  |
| miniImageNet-C 1-shot 8-target   | 64.27                  |
| miniImageNet-C 1-shot 16-target  | 75.22                  |
| FEMNIST-FewShot 1-shot 1-target  | 99.72                  |