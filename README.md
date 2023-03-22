WMH MRI Segmentation
====================

The data should be already downloaded and extracted in the `data_root` folder. For more info on the data, see the [data README](https://www.notion.so/matzkin/2-WMH-Data-download-bf662e9460c444459e3934d3099d9285).

The hyperparameters are set in the MLproject file. The hyperparameters are:

**General parameters**
- data_root: the root directory of the data

**Training parameters**
- train_centers: the centers used for training in the format split1:center1,center2;split2:center1,center2, where splits: [training, test] and centers: [Utrecht, Amsterdam, Singapore]
- batch_size: the batch size
- epochs: the number of epochs
- learning_rate: the learning rate
- weight_decay: the weight decay
- momentum: the momentum

**Testing parameters**
- test_batch_size: the batch size for testing
