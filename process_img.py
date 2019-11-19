## input.py
## Process input images (data) to fit the requirements of the AI

## DONWLOAD CURRENT DATASET? See dropbox paper links (ask Twan)

import os # File navigation
import cv2 # Image manipulation
import numpy as np # Data manipulation
from tqdm import tqdm # Pretty progressbars :)
import sys

# Setup
np.set_printoptions(threshold=sys.maxsize)


# Speciffy if you want to rebuilt the data yes(=True), or not(=False)
# TODO: Replace wit commandline argument (no hardcoding)
REBUILT_DATA = False

class FaceData():

    #----! GENERAL IMAGE SPECS !----#
    # The size of the images (IMG_SIZExIMG_SIZE) [px]
    # TODO: Replace wit cmd-line arg 
    IMG_SIZE = 50

    #----! FOLDERS AND LABELS !----#
    # Faces root directories (RAW)
    # TODO: Replace wit cmd-line arg 
    ROOT_DIR = 'faces'

    # Save format
    file_save_name = 'training_data_{}'

    # LABELS
    FACES_LABELS = [] 
    # FACES_LABELS_NAME = []
    # Counter variables
    # To keep track of the there is generally an equal amount of pictures per label
    max_pictures = 15 # An educated guess (based on knowledge of availible data)
    min_pictures = 5

    # Keep track of amnt of labels (for double checking)
    n_labels = 0

    #----! TRAINING DATA !----#
    training_data = []


    def gather_folders_and_labels(self):

        # Make scoped copy of self.ROOT_DIR        
        # Also convert it to a os dir
        root_dir = self.ROOT_DIR
        folder_paths = [] # Keep track of the folder paths
        # Get all folders in the root_dir
        for folder in os.listdir(root_dir):
            # Create the exact folder path
            folder_path = os.path.join(os.path.abspath('.'), root_dir, folder)
            # Check if 'folder' is an actual folder
            if os.path.isdir(folder_path):
                # Check if there are enough pictures 
                if len(os.listdir(folder_path)) >= self.min_pictures:
                    # Count the number of labels (=folders)
                    self.n_labels += 1
                    # Create the label string 
                    label = folder[5:].lower().replace(' ', '_')
                    folder_paths.append(folder_path)
                    self.FACES_LABELS.append(label)                    
        for indx, path in enumerate(folder_paths):
            # Loop trough files in the folder  
            # Limit at self.max_pictures
            for file in os.listdir(path)[:self.max_pictures]:
                # Construct abs file path
                file_path = os.path.join(path, file)
                # Check if it is an file
                if os.path.isfile(file_path):
                    try:
                        # Load image in grayscale
                        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        # Resize to IMG_SIZEz
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # Add label
                        label = self.FACES_LABELS[indx]
                        # Add to training_data
                        self.training_data.append([np.array(img), np.eye(self.n_labels)[self.FACES_LABELS.index(label)]])
                    except Exception as e:
                        del self.FACES_LABELS[indx]
                        print(str(e))
    
        np.random.shuffle(self.training_data)
        return [self.training_data, self.FACES_LABELS]
        
                        





        
fd = FaceData()
td = fd.gather_folders_and_labels()
np.save('training_data.npy', td)

