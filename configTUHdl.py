# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications
data_folders = ['E:/NMT_dataset/normal','E:/NMT_dataset/abnormal']
processed_folder='D:/NMT_processed'
n_recordings = None  # set to an integer, if you want to restrict the set size
sensor_types = ["EEG"]
n_chans = 21
max_recording_mins = None # exclude larger recordings from training set
duration_recording_mins = 2  # how many minutes to use per recording
max_abs_val = 800  # for clipping
sampling_freq = 250
divisor = 10  # divide signal by this
test_on_eval = True  # test on evaluation set or on training set
shuffle = True

input_time_length = 15000
model_constraint = 'defaultnorm'
init_lr = 0.000625
batch_size = 16
max_epochs = 35 # until first stop, the continue train on train+valid
cuda = True # False
