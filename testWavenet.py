import numpy as np
import pandas as pd
import torch
import h5py
import skorch
from torch.utils.data import DataLoader
from skorch.helper import predefined_split
from config import *
from dataset import *
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mne import set_log_level
import os
from WavenetLSTM import WavenetLSTM,WavePathModel,LSTMPathModel
from sklearn.metrics import recall_score
def sensitivity(net, ds, y=None):
    # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
    y_true = [y for _, y in ds]
    y_pred = net.predict(ds)
    return recall_score(y_true, y_pred)

def specificity(net, ds, y=None):
    # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
    y_true = [y for _, y in ds]
    y_pred = net.predict(ds)
    return recall_score(y_true, y_pred,pos_label=0)
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, excel_path):
        super().__init__()
        excel_file=pd.read_excel(excel_path)
        self.file_names=excel_file['name'].to_numpy(dtype=str)
        self.label=excel_file['label'].to_numpy()

    def __getitem__(self, index):
        with h5py.File(self.file_names[index], 'r') as h5_file:
            window=np.array(h5_file['x'])

        label=self.label[index]
        return window,label
 
    def __len__(self):
        return len(self.label)
    
if __name__=='__main__':
    set_log_level(False)
    torch.set_anomaly_enabled(True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    device = 'cuda' if cuda else 'cpu'
    torch.backends.cudnn.benchmark = True
    test_set=WindowDataset(f'{processed_folder}/eval.xlsx')
    train_set=WindowDataset(f'{processed_folder}/train.xlsx')

    #Model initialization
    if 'wavenetLSTM' == model_name:
        model=WavenetLSTM()
    elif 'wavenet' == model_name:
        model=WavePathModel()
    elif 'lstm' == model_name:
        model=LSTMPathModel()

    model_name+='_'+processed_folder
    
    print(model_name)
    
    classifier = skorch.NeuralNetClassifier(
            model,
            train_split=predefined_split(test_set),
            iterator_valid=DataLoader,
            iterator_valid__pin_memory=True,
            iterator_valid__num_workers=4,
            iterator_valid__persistent_workers=True,
            batch_size=batch_size,
            device=device,
            classes=[0,1],
            warm_start=True,
            )
    classifier.initialize()
    classifier.load_params(f_params=f'model/{model_name}best_param.pkl')
    
    print("Paramters Loaded")
    pred_labels=classifier.predict(test_set)
    actual_labels=[label[1] for label in test_set]
    print('Sensitivity',recall_score(actual_labels,pred_labels))
    conf_mat=ConfusionMatrixDisplay.from_predictions(actual_labels,pred_labels)
    conf_mat.plot()
    plt.show()