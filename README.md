WMH MRI Segmentation
====================

The data should be already downloaded and extracted in the `data_root` folder. 
For more info on the data, see the [data README](https://www.notion.so/matzkin/2-WMH-Data-download-bf662e9460c444459e3934d3099d9285).

The hyperparameters are set in the MLproject file. The hyperparameters are:

## Training

Note that `mlflow run` is called with `--env-manager=local` to avoid reinstalling the environment each time. The dependencies can be automatically installed omitting this parameter.

### Utrecht
For training with utrecht data, run:

#### Loss: DiceCE

```
mlflow run . --env-manager=local
```

#### Loss: Focal

```
mlflow run . -P loss=focal --env-manager=local
```

#### Loss: CE + MEEP

```
mlflow run . --env-manager=local
```

#### Loss: Dice + MEEP

```
mlflow run . --env-manager=local
```

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