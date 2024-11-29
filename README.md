Recreation of Automatic Detection of Abnormal EEG Signals Using WaveNet and LSTM (https://doi.org/10.3390/s23135960).
Base Wavenet model taken from https://github.com/golbin/WaveNet and base CAIN taken from https://github.com/myungsub/CAIN

To train the models, follow these steps:
1- modify the config.py file according to your configurations i.e. dataset location and model to be trained
2- run create_dataset.py to preprocess the dataset into hdf5 format
3- run trainWavenet.py to train the selected model
4- run testWavenet.py to test the selected model and get confusion matrix