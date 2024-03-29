## train.py
## Train the neural network (from net.py) with the training data (V2 or V1)

#----! IMPorTS !----#

# Py torch nueral network models/funcs
import torch
import torch.nn as nn
import torch.optim as optim

# Pretty progress bars
from tqdm import tqdm

# Data manipulation/storage
import numpy as np
# Get the facenet
from net import FaceNet

#----! SETUP !----#

# Initilizate FaceNet model def
face_net = FaceNet()

# Get the training data
# TODO: CMD arg
training_data_file_name = 'julian_twan_training_data_1575409409.1191235.npy' # CHANGE TO THE DATA YOu WANT TO USE
file_data = np.load(training_data_file_name, allow_pickle=True)
print(file_data)
training_data = file_data


## Create the optimizer
optimizer = optim.Adam(face_net.parameters(), lr=0.01)
loss_function = nn.MSELoss()

# Get the X, y from the training data
# Also make them into a tensor
# X: Data
# y: labels

# i[0] is the image data
# i[1] is the coresponding result (aka label) list/array
X = torch.Tensor([i[0] for i in file_data]).reshape(-1,50,50)# Reshape to 50x50
X = X/255.0  # Nomraly pixel values are between 0-255 we want them to be between 0-1 (performace, ease of learning)
y = torch.Tensor([i[1] for i in training_data])

# Split training/testing data
VAL_PCT = 0.1 #TODO: Remove hardcode 
val_size = int(len(X)*VAL_PCT)

print(val_size) #Debug

# Create the training and testing data portions by slicing them
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

#### TRaINING !!!!
# The amount of images being processed at once, depends on availble memoery
BATCH_SIZE = 100 # TODO: No hardcoding
# Round of learning
EPOCHS = 1 #TODO: No hardcoding

# Loop over the amount of epochs
for epoch in range(EPOCHS):
    # Loop over the data
    # range(start, end, step_size)
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # Create batches
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]
        print(batch_X.shape)
        print(batch_y.shape)
        print(batch_y)
        
        # Zero the gradients for optimal learning and no bias
        optimizer.zero_grad()
        outputs = face_net(batch_X)
        print(outputs.shape)
        # loss = loss_function(outputs, batch_y)
        # loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}\n")

#print(loss) # Debug

### TESTING

correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = face_net(test_X[i].view(-1, 1, 50, 50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class: 
            correct += 1
        
        total += 1
        
print("Acc:", round(correct/total, 3))