from torch import nn
from math import floor

def getOutputSize(input_size, kernel_size, stride, padding=0,dilation=1):
    """
    Calculate the output size of a convolutional layer
    """
    return floor(((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)
class ResNet(nn.Module):
    """
        ResNet model
        -Conv1D
        -BatchNorm1D
        -ReLU
        -4 ResBlocks
        -Dense Layer
        -Sigmoid
    """
    def __init(self,in_channels,out_channels):
        
        super().__init__()
        self.Conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm1 = nn.BatchNorm1d(out_channels)
        self.ReLU1 = nn.ReLU()
        resblocks = []
        for i in range(4):
            double_channels = out_channels*2
            resblocks.append(ResBlock(out_channels, double_channels))
            out_channels = double_channels
        self.ResBlocks = nn.Sequential(*resblocks)
        
        
    def forward(self, x):
        pass
    
class ResBlock(nn.Module):
    """
        Residual block:
        - One copy of the input goes through MaxPool1D and 1x1 Convolutions
        - The other copy goes through Conv1D, BatchNorm1D, ReLU, Dropout and Conv1D
        - The two copies are added together (so the shapes must match)
        - The result goes through BatchNorm1D, ReLU and Dropout
    """
    def __init__(self, in_channels, out_channels):    
        super().__init__()
        self.MaxPool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.Conv1_A = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.Conv1_B = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2)
        self.BatchNorm1 = nn.BatchNorm1d(out_channels)
        self.ReLU1 = nn.ReLU()
        self.Dropout1 = nn.Dropout()
        self.Conv2_B = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2 = nn.BatchNorm1d(out_channels)
        self.ReLU2 = nn.ReLU()
        self.Dropout2 = nn.Dropout()
    def forward(self, x):
        y = x
        x = self.MaxPool1(x)
        #1999x1999 A
        x = self.Conv1_A(x)
        #1999x1999 A
        y = self.Conv1_B(y)
        #1999x1999 B
        y = self.BatchNorm1(y)
        #1999x1999 B
        y = self.ReLU1(y)
        #1999x1999 B
        y = self.Dropout1(y)
        #1999x1999 B
        y = self.Conv2_B(y)
        # OKAY SAME SHAPE
        x = x + y
        x = self.BatchNorm2(x)
        x = self.ReLU2(x)
        x = self.Dropout2(x)
        return x
    