import numpy as np
import pandas as pd
import torch
import h5py
import skorch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.callbacks import Checkpoint
from skorch.helper import predefined_split
from configTUHdl import *
from dataset import *
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mne import set_log_level
from skorch.callbacks import LRScheduler,EarlyStopping,EpochScoring
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
    model_name='lstm'
    criterion=torch.nn.CrossEntropyLoss
    optimizer_lr=0.000025

    #Model initialization
    if 'wavenetLSTM' == model_name:
        model=WavenetLSTM()
    elif 'wavenet' == model_name:
        model=WavePathModel()
    elif 'lstm' == model_name:
        model=LSTMPathModel()

    if 'NMT' in processed_folder:
        model_name+='_NMT'
    
    print(model_name)
    monitor = lambda net: any(net.history[-1, ('valid_accuracy_best','valid_f1_best','valid_loss_best')])
    cp=Checkpoint(monitor='valid_acc_best',dirname='model',f_params=f'{model_name}best_param.pkl',
               f_optimizer=f'{model_name}best_opt.pkl', f_history=f'{model_name}best_history.json')
    scheduler=LRScheduler(policy=ReduceLROnPlateau,monitor='train_loss',factor=0.1,patience=2)
    
    classifier = skorch.NeuralNetClassifier(
            model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(test_set),
            optimizer__lr=optimizer_lr,
            #optimizer__weight_decay=optimizer_weight_decay,
            iterator_train=DataLoader,
            iterator_valid=DataLoader,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            iterator_valid__pin_memory=True,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            iterator_train__persistent_workers=True,
            iterator_valid__persistent_workers=True,
            batch_size=batch_size,
            device=device,
            callbacks=[('train_acc', EpochScoring(
                'accuracy',
                name='train_acc',
                lower_is_better=False,
            )),
            ('valid_sensitivity', EpochScoring(
                sensitivity,
                name='valid_sensitivity',
                lower_is_better=False,
            )),
            ('valid_specificity', EpochScoring(
                specificity,
                name='valid_specificity',
                lower_is_better=False,
            )),
            cp,skorch.callbacks.ProgressBar(), scheduler
            #, EarlyStopping(patience=10),
            ],
            warm_start=True,
            )
    classifier.initialize()
    classifier.load_params(
        f_params=f'model/{model_name}best_param.pkl', f_history=f'model/{model_name}best_history.json')
    
    print("Paramters Loaded")
    pred_labels=classifier.predict(test_set)
    actual_labels=[label[1] for label in test_set]
    print(pred_labels)
    #print((actual_labels==1)==(pred_labels==1))
    print('Sensitivity',recall_score(actual_labels,pred_labels))
    conf_mat=ConfusionMatrixDisplay.from_predictions(actual_labels,pred_labels)
    conf_mat.plot()
    plt.show()