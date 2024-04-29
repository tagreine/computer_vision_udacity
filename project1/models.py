## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
try:
    from torchsummary import summary
except:
    print("No module named torchsummary")

class Net(nn.Module):

    def __init__(self, width, height):
        super(Net, self).__init__()
        
        self.width = width
        self.height = height
        assert width % 16 == 0 and height % 16 == 0, "width and height should be divisible by 16"
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # creating a modified mini-version of the network in Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet https://doi.org/10.48550/arXiv.1710.00977 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2)    
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                      
        #self.drop1 = nn.Dropout(0.1) # I do think that drop out in the cnn layers should be removed, since pixels are strongly correlated in a cnn
        self.bchn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)   
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                           
        #self.drop2 = nn.Dropout(0.1)
        self.bchn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                           
        #self.drop3 = nn.Dropout(0.1)
        self.bchn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0) 
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                           
        self.drop4 = nn.Dropout(0.1)   
        self.dens1 = nn.Linear(in_features= 64 * (self.width * self.height) // (16 * 16) , out_features=1000) 
        self.drop5 = nn.Dropout(0.2)   
        self.dens2 = nn.Linear(in_features=1000, out_features=1000) 
        self.drop6 = nn.Dropout(0.2)  
        self.dens3 = nn.Linear(in_features=1000, out_features=136) 


        ## Note that among the layers to add, consider including:
                # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
    
    @property   
    def print_summary(self):
        print(summary(Net(self.width, self.height)))

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.bchn1(self.pool1(F.relu(self.conv1(x))))
        #print(x.shape)
        x = self.bchn2(self.pool2(F.relu(self.conv2(x))))
        #print(x.shape)
        x = self.bchn3(self.pool3(F.relu(self.conv3(x))))
        #print(x.shape)        
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.drop5(F.relu(self.dens1(x)))
        x = self.drop6(F.relu(self.dens2(x)))
        x = self.dens3(x)

        return x

