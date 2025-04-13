WMH MRI Segmentation
====================

The data should be already downloaded and extracted in the `data_root` folder.

The default hyperparameters are set in the MLproject file.

## Environment Setup

You have two options to set up the environment:

### Option 1: Using Docker (Recommended)

1. Install [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/install/)

2. Edit the `docker-compose.yaml` file to map your datasets directory:
  ```yaml
  volumes:
    - .:/app
    - jupyter-data:/home/appuser/.local/share/jupyter
    - /path/to/your/datasets:/home/appuser/Code/datasets
  ```

3. Start the Docker container:
  ```
  docker-compose up
  ```

4. Connect to the Jupyter server:
  - Open the URL displayed in the terminal (typically http://localhost:8888/?token=...)
  - When running notebooks in VSCode:
    - Open a notebook file
    - Click on the kernel selector in the top-right corner
    - Choose "Select Kernel..." or "Select Another Kernel..."
    - Select "Jupyter Server..." and enter the URL from the terminal
    - Choose the Python kernel from the Docker container

### Option 2: Manual Setup

If you use mlflow, the environment will be created automatically. Otherwise, you can create it manually with:

```
python3.10 -m venv env-wmh-mc-seg
source env-wmh-mc-seg/bin/activate  # On Windows: env-wmh-mc-seg\Scripts\activate
pip install -r requirements.txt
```

It should be activated before running any of the scripts.

## Training
The models can be trained both using the mlflow CLI or calling the src/train.py script. The only difference is that for the CLI, the run name (for mlflow tracking) is set automatically by mlflow, while for the script, it's set according to the centers used for train and the loss function used.

  
  For both alternatives, the parameters will be taken from the MLProject file first, and the ones passed as arguments will set/override them.

### With train.py script
Located in the project folder (where the MLproject file is located), run:
```
python src/train.py --centers='training:Singapore' --loss='KL'
```

### With MLflow CLI
Located in the project folder (where the MLproject file is located), run:
```
mlflow run . -P centers='training:Singapore' -P loss='KL' --env-manager=local --run-name='training_Singapore_KL'
```

Note that `mlflow run` is called with `--env-manager=local` to prevent the download/installation of the environment each time. The dependencies can be automatically installed omitting this parameter.

Also, since the run name is set automatically by mlflow, the --run-name parameter can be specified (or not) to override it.

Refer to the [MLflow docs](https://www.mlflow.org/docs/latest/cli.html#mlflow-run) for more info on the CLI.

## Predict
### With predict.py script

Located in the project folder (where the MLproject file is located), run:
```
python src/predict.py --centers='test:Singapore' --model_path='checkpoints/training_Singapore_best.ckpt' --mc_samples=10 --mc_ratio=0.1
```

### With MLflow CLI

Located in the project folder (where the MLproject file is located), run:
```
mlflow run . -e predict -P centers='test:Singapore' -P model_path='checkpoints/training_Singapore_best.ckpt' -P mc_samples=10 -P mc_ratio=0.1 --env-manager=local
```


## Parameters

The parameters are set in the MLproject file and have default values. Some of them could require tuning:

- **train_centers:** Centers used in the format split1:center1,center2;split2:center1,center2, where splits: [training, test] and centers: [Utrecht, Amsterdam, Singapore]
- **seed:** Random seed used for splitting the data.
- **split_ratios:** List of ratios for train/validation/test splits, for the training images. The sum of the ratios should be 1.0. Note that this parameter can be used in the test phase as well.

   This parameter is used along with the seed parameter. Each training partition used in train_centers, will be split separately according to this seedn and in case there's more than one center used, then will be merged.
- **samples_per_volume**: How many patches to take from each volume.
- **queue_length**: Amount of patches loaded in memory for online processing. See [TorchIO docs](https://torchio.readthedocs.io/_modules/torchio/data/queue.html).
- **tio_num_workers**: Number of subprocesses to use for data loading. See [TorchIO docs](https://torchio.readthedocs.io/_modules/torchio/data/queue.html).
- **mc_ratio**: Ratio of dropout to use for Monte Carlo dropout estimation during inference. This ratio should be lower than the dropout used in training.
- **mc_samples**: Number of samples to use for Monte Carlo dropout estimation.

### Default hiperparameters for training

| Parameter | Value | 
| --- | --- |
| data root | ~/Code/datasets/wmh |
| centers | training:Utrecht |
| split ratios | [0.7, 0.1, 0.2] |
| epochs | 800 |
| batch size | 64 |
| lr | 0.001 |
| dropout | 0.2 |
| loss | MEEP |
| weight decay | 0 |
| seed | 42 |
| patch size | 32 |
| samples per volume | 20 |
| queue length | 500 |
| tio num workers | 8 |
| reg start | 0 |
| meep lambda | 0.3 |

Run `python src/train.py --help` for more info on the parameters.