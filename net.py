import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceNet(nn.Module):
    '''
    FaceNet
    * The neural network that will learn to reconize face
    '''
    def __init__(self):
        '''
        __init__
        * Take care of initization of the model
        * Layout the layers
        '''

        # Initlizalize the model 
        super(FaceNet, self).__init__()

        #----! LAYERS !----#

        # First convelutional layer
        # (1, 32, 5) -> (input, output, kernel_size) output=#convelutional features kernal_size=5x5 kernal (window) as it goes over the features
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Seccond layer
        # input is now 32 from prev layer
        self.conv2 = nn.Conv2d(32, 64, 5)        
        # Third layer
        self.conv3 = nn.Conv2d(64, 128, 5)           

        # Prepping for output layer
        # We do not know the exact shape of the output so we need to calc it
        x = torch.randn(50,50).view(-1, 1, 50, 50) # Random tensor from 50x50 reshaped to 1x50x50 
        self._to_linear = None
        self.convs(x)
        
        # Output layers (fully conected layer)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2) # 2 for the amount of classes

    

    def convs(self, x):
        # Only going to run the first 3 conv layers
        # This method is used to calc the shape of the output from the convelutional layers 
        # Can also be used in the formward() method 
        
        # Max_pool = ? relu=rectified linear (2,2)=shape of the pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        #print(x[0].shape)
        if self._to_linear is None:
            # The shape of the output 
            # x is 'all' the data, so we use the first element 
            ## keep in mind in this case (_to_linear=None) that x is 'fake' randomly generated data
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            #print(self._to_linear)
        
        return x

    def forward(self, x):
        # The 'actual' forward method
        
        # Run convs (DRY)
        x = self.convs(x)
        # Flatten x
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

