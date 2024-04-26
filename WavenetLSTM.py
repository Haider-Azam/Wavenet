from wavenet import WaveNet as WaveNetModel
from cain import CAIN
import torch
from torch.nn import AvgPool1d,LSTM,Dropout,Linear,Tanh,Flatten,Softmax

class WavePath(torch.nn.Module):
    def __init__(self):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(WavePath, self).__init__()
        self.pool1=AvgPool1d(kernel_size=10,stride=10)
        self.pool2=AvgPool1d(kernel_size=2,stride=2)

        self.waveblock1=WaveNetModel(layer_size=5,stack_size=1,in_channels=20,res_channels=16,dilation_rate=8,kernel_size=3)
        self.waveblock2=WaveNetModel(layer_size=5,stack_size=1,in_channels=16,res_channels=32,dilation_rate=5,kernel_size=3)
        self.waveblock3=WaveNetModel(layer_size=5,stack_size=1,in_channels=32,res_channels=64,dilation_rate=3,kernel_size=3)
        self.waveblock4=WaveNetModel(layer_size=3,stack_size=1,in_channels=64,res_channels=64,dilation_rate=2,kernel_size=2)

        self.lstm=LSTM(input_size=64,hidden_size=64,num_layers =3,batch_first=True)

        self.dropout=Dropout(p=0.5)
    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, channels, timestep]
        :return: Tensor[batch, channels, timestep]
        """
        output=self.waveblock1(x)
        
        output=self.pool1(output)

        output=self.waveblock2(output)
        
        output=self.pool1(output)

        output=self.waveblock3(output)
        
        output=self.pool1(output)
        
        output=self.waveblock4(output)

        output=self.pool2(output)

        output,_=self.lstm(output.mT)

        output=output[:,-1,:]

        output=self.dropout(output)
        
        return output
    
class LSTMPath(torch.nn.Module):
    def __init__(self,window_size=500,input_size=20,in_channels=30):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(LSTMPath, self).__init__()
        self.window_size=window_size
        self.input_size=input_size
        self.in_channels=in_channels
        
        self.lstm1=LSTM(input_size=input_size,hidden_size=input_size,num_layers = 3,batch_first=True)

        self.lstm2=LSTM(input_size=input_size,hidden_size=input_size,num_layers = 3,batch_first=True)

        self.ch_attention=CAIN(depth=2,in_channels=in_channels)

        self.dropout=Dropout(p=0.2)

        self.dense=Linear(in_features=20,out_features=2)

        self.tanh=Tanh()

        self.flatten=Flatten()
    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, channels, timestep]
        :return: Tensor[batch, channels, timestep]
        """
        assert x.dim()==3

        #Reversing the timestep
        output=torch.flip(x,[-1]).mT

        no_of_windows = output.shape[1]//self.window_size

        windowed_input = output.contiguous().view(-1,no_of_windows,self.window_size,self.input_size)

        lstm_input = windowed_input.contiguous().view(-1,self.window_size,self.input_size)

        output, _ = self.lstm1(lstm_input)

        output = output[:,-1,:]
        
        output = output.contiguous().view(x.size(0),no_of_windows,self.input_size)

        output, feats = self.ch_attention(output,output)

        output, _ = self.lstm2(output)

        output = self.dropout(output)

        output = self.dense(output)

        output = self.tanh(output)

        output = self.flatten(output)
        return output
    

class WavenetLSTM(torch.nn.Module):
    def __init__(self,input_time_length=15000,window_size=500,input_size=20):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(WavenetLSTM, self).__init__()
        self.input_time_length=input_time_length
        self.window_size=window_size
        self.input_size=input_size

        self.cain_channels=input_time_length//window_size
        
        self.wave_path=WavePath()

        self.lstm_path=LSTMPath(window_size=window_size,input_size=input_size,in_channels=self.cain_channels)

        self.dense=Linear(in_features=124,out_features=2)

        self.softmax=Softmax(dim=1)
    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, channels, timestep]
        :return: Tensor[batch, channels, timestep]
        """
        assert x.dim()==3

        wave_output = self.wave_path(x)

        lstm_output = self.lstm_path(x)

        output = torch.cat([wave_output, lstm_output], dim=1)

        output = self.dense(output)

        output = self.softmax(output)   
        return output