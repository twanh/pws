#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os 
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50 # Images will be 50x50

    # Directories of the images
    CATS = "cnn-img/PetImages/Cat"
    DOGS = "cnn-img/PetImages/Dog"    
    
    # Labels (classes)
    LABELS = {CATS: 0, DOGS: 1}
    
    training_data = []
    
    # Counters (count training samples) -- for balancing the data
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        # Loop over the folders 
        for label in self.LABELS:
            print(label)
            # Loop over files in the folders
            for f in tqdm(os.listdir(label)):
                try:
                    # Join the paths together
                    path = os.path.join(label, f)
                    # Load img, and convert to grayscale
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Resize to IMG_SIZE
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # Add to training data
                    # np.eye(len_of_vector)[indx_to_be_true]
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # Add to counter
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    #print(str(e))
                    pass
    
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)
        
        
if REBUILD_DATA:
    dogvscats = DogsVSCats()
    dogvscats.make_training_data()
    
training_data = np.load('training_data.npy', allow_pickle=True)
print(len(training_data))


# In[33]:


import torch 
import torch.nn as nn
import torch.nn.functional as F


# Outline the neural network (CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Layers
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
    
    
net = Net()
print(net)


# In[26]:


import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


# Seperate X,y from training_data
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) # Reshape to 50x50
X = X/255.0  # Nomraly pixel values are between 0-255 we want them to be between 0-1 (performace, ease of learning)
y = torch.Tensor([i[1] for i in training_data])


# Split training/testing data
# Percentage of testing (validation=val) data
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)


# In[29]:


train_X = X[:-val_size] # Slicing
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]
print(len(train_X))
print(len(train_y))


# In[34]:


BATCH_SIZE = 100 # Depents on availible memory
EPOCHS = 1 # Rounds of learning

for epoch in range(EPOCHS):
    # range(start, end, steps)
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # Created batches
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]
        
        # Zero gradients
        optimizer.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
print(loss)
        


# In[36]:


correct = 0
total = 0


with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class: 
            correct += 1
        
        total += 1
        
print("Acc:", round(correct/total, 3))

