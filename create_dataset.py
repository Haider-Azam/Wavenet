import numpy as np
import h5py
import pandas as pd
from config import *
from dataset import *
from mne import set_log_level
import resampy
from scipy.signal import butter,iirnotch,filtfilt
set_log_level(False)
device = 'cuda' if cuda else 'cpu'
ch_names=['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1','FP2', 'FZ',
               'O1', 'O2','P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

#Implementing Transverse Central Parietal (TCP) montage technique for a single sample
def tcp(data,fs):
    length=data.shape[1]
    new_data =np.zeros(shape=(20,length),dtype=np.float32)
    
    new_data[0] = (data[ch_names.index('FP1'),:]) - (data[ch_names.index('F7'),:])
    new_data[1] = (data[ch_names.index('FP2'),:]) - (data[ch_names.index('F8'),:])
    new_data[2] = (data[ch_names.index('F7'),:]) - (data[ch_names.index('T3'),:])
    new_data[3] = (data[ch_names.index('F8'),:]) - (data[ch_names.index('T4'),:])
    new_data[4] = (data[ch_names.index('T3'),:]) - (data[ch_names.index('T5'),:])
    new_data[5] = (data[ch_names.index('T4'),:]) - (data[ch_names.index('T6'),:])
    new_data[6] = (data[ch_names.index('T5'),:]) - (data[ch_names.index('O1'),:])
    new_data[7] = (data[ch_names.index('T6'),:]) - (data[ch_names.index('O2'),:])
    new_data[8] = (data[ch_names.index('T3'),:]) - (data[ch_names.index('C3'),:])
    new_data[9] = (data[ch_names.index('T4'),:]) - (data[ch_names.index('C4'),:])
    new_data[10] = (data[ch_names.index('C3'),:]) - (data[ch_names.index('CZ'),:])
    new_data[11] = (data[ch_names.index('CZ'),:]) - (data[ch_names.index('C4'),:])
    new_data[12] = (data[ch_names.index('FP1'),:]) - (data[ch_names.index('F3'),:])
    new_data[13] = (data[ch_names.index('FP2'),:]) - (data[ch_names.index('F4'),:])
    new_data[14] = (data[ch_names.index('F3'),:]) - (data[ch_names.index('C3'),:])
    new_data[15] = (data[ch_names.index('F4'),:]) - (data[ch_names.index('C4'),:])
    new_data[16] = (data[ch_names.index('C3'),:]) - (data[ch_names.index('P3'),:])
    new_data[17] = (data[ch_names.index('C4'),:]) - (data[ch_names.index('P4'),:])
    new_data[18] = (data[ch_names.index('P3'),:]) - (data[ch_names.index('O1'),:])
    new_data[19] = (data[ch_names.index('P4'),:]) - (data[ch_names.index('O2'),:])

    return new_data, fs

#HDF5 implementation
def create_hdf5(processed_folder,data,label,split):
    file_names=[]
    path=os.path.join(processed_folder,split)
    for i in range(len(label)):
        file_path=f'{path}/{i}.hdf5'
        file_names.append(file_path)
        with h5py.File(file_path, 'a') as f:
            f['x']=data[i]
            f['y']=label[i]

    file_names=pd.Series(file_names)
    lbs=pd.Series(label)
    dataframe=pd.DataFrame({'name':file_names,'label':lbs})
    dataframe.to_excel(f"{processed_folder}/{split}.xlsx",index=False)

if __name__=='__main__':
    butter_b,butter_a=butter(4,1,btype='highpass',fs=sampling_freq)
    notch_b,notch_a=iirnotch(60,Q=30,fs=sampling_freq)
    preproc_functions = []
    #Cut to 2 minutes length
    preproc_functions.append(lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))
    #Apply butterworth and notch filter
    preproc_functions.append(lambda data, fs: (filtfilt(butter_b, butter_a, data), fs))
    preproc_functions.append(lambda data, fs: (filtfilt(notch_b, notch_a, data), fs))
    #Apply TCP montage technique
    preproc_functions.append(tcp)
    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,sampling_freq,axis=1,filter='kaiser_fast'),sampling_freq))

    dataset = DiagnosisSet(n_recordings=n_recordings,
                            max_recording_mins=max_recording_mins,
                            preproc_functions=preproc_functions,
                            data_folders=data_folders,
                            train_or_eval='train',
                            sensor_types=sensor_types)
    if test_on_eval:
        test_dataset = DiagnosisSet(n_recordings=n_recordings,
                            max_recording_mins=max_recording_mins,
                            preproc_functions=preproc_functions,
                            data_folders=data_folders,
                            train_or_eval='eval',
                            sensor_types=sensor_types)
        
    X,y=dataset.load()
    test_x,test_y=test_dataset.load()

    train_data=np.array(X)
    test_data=np.array(test_x)
    del X,test_x

    augmented_train_data=np.concatenate([train_data[:,:,:input_time_length] , train_data[:,:,-1:input_time_length-1:-1]])
    augmented_test_data=np.concatenate([test_data[:,:,:input_time_length] , test_data[:,:,-1:input_time_length-1:-1]])

    augmented_train_label=np.concatenate([y,y])
    augmented_test_label=np.concatenate([test_y,test_y])
    
    create_hdf5(processed_folder,augmented_train_data,augmented_train_label,'train')
    create_hdf5(processed_folder,augmented_test_data,augmented_test_label,'eval')