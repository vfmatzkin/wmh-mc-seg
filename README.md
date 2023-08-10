WMH MRI Segmentation
====================

The data should be already downloaded and extracted in the `data_root` folder. 
For more info on the data, see the [data README](https://www.notion.so/matzkin/2-WMH-Data-download-bf662e9460c444459e3934d3099d9285).

The default hyperparameters are set in the MLproject file.

[//]: # (## Quickstart)

[//]: # (### Get predictions from pretrained models)

[//]: # (#### White Matter Intensity segmentation masks)

[//]: # (#### Uncertainty estimates)

[//]: # (##### Using entropy)

[//]: # (##### Using MC dropout)


## Environment creation

If you use mlflow, the environment will be created automatically. Otherwise, you can create it manually with:

```
python3.10 -m venv env-wmh-mc-seg
source env-wmh-mc-seg/bin/activate
pip install -r requirements.txt
```

## Training

### Some considerations
- Note that `mlflow run` is called with `--env-manager=local` to prevent the download/installation of the environment each time. The dependencies can be automatically installed omitting this parameter.
- The models can be trained both using the mlflow CLI or calling the src/train.py script. The only difference is that for the CLI, the run name (for mlflow tracking) is set automatically by mlflow, while for the script, it's set according to the centers used for train and the loss function used.

  This is why you could set the --run-name parameter in the CLI with a custom name.
  
  For both alternatives, the parameters will be taken from the MLProject file first, and the ones passed as arguments will set/override them.

### Utrecht
For training with utrecht data, run:

#### Loss: DiceCE

```
mlflow run . -P centers='training:Utrecht' --env-manager=local --run-name=training_Utrecht_DiceCE
```

#### Loss: Focal

```
mlflow run . -P centers='training:Utrecht' -P loss=focal --env-manager=local
```

#### Loss: CE + MEEP

```
mlflow run . -P centers='training:Utrecht' -P loss=MEEP -P meep_start=50 -P meep_lambda=0.3 --env-manager=local
```

Note that when choosing this loss, the `meep_start` and `meep_lambda` parameters should be set. The `meep_start` parameter indicates the epoch from which the MEEP loss will be used. The `meep_lambda` parameter indicates the weight of the MEEP loss. Check default values in the MLproject file otherwise.

### Amsterdam

For training with amsterdam data, run:

```
mlflow run . -P centers='training:Amsterdam' --env-manager=local
```
### Singapore

For training with singapore data, run:

```
mlflow run . -P centers='training:Singapore' --env-manager=local
```

## Predict

### Some considerations

- If you're using Monte Carlo dropout estimation (setting the options `--mc-ratio` and `--mc-samples`), `--mc-ratio` should be lower than the `dropout` option used for training.

### Utrecht

```
mlflow run . -e test --env-manager=local
```


### Amsterdam

```
mlflow run . -e test -P model_path=~/Code/wmh-mc-seg/checkpoints/training_Amsterdam_best.ckpt --env-manager=local
```

### Singapore

```
mlflow run . -e test -P model_path=~/Code/wmh-mc-seg/checkpoints/training_Singapore_best.ckpt --env-manager=local
``` 


**Parameters**
The parameters are set in the MLproject file and have default values. Some of them which could require tuning are:

- **train_centers:** Centers used in the format split1:center1,center2;split2:center1,center2, where splits: [training, test] and centers: [Utrecht, Amsterdam, Singapore]
- **seed:** Random seed used for splitting the data.
- **split_ratios:** List of ratios for train/validation/test splits, for the training images. The sum of the ratios should be 1.0. Note that this parameter can be used in the test phase as well.

    This parameter is used along with the seed parameter. Each training partition used in train_centers, will be split separately according to this seedn and in case there's more than one center used, then will be merged.
- **samples_per_volume**: How many patches to take from each volume.
- **queue_length**: Amount of patches loaded in memory for online processing. See [TorchIO docs](https://torchio.readthedocs.io/_modules/torchio/data/queue.html).
- **tio_num_workers**: Number of subprocesses to use for data loading. See [TorchIO docs](https://torchio.readthedocs.io/_modules/torchio/data/queue.html).