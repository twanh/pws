## process_img_v2.py
## V2
## Process input images into the right shape and convert them into an array
## Processes only the faces from Julian & Twan to allow training on these two


## SEE DATA IN PWS GOOGLE DRIVE FOR IMAGES
## Folder structure
## - julian_twan/  || Contains all the images
##  - julian/      || Contains the images of Julian
##  - twan/        || Contains the images of Twan

#----! IMPoRTS !----#

import os # File and folder navigation
import cv2 # Image manipulation
import numpy as np # Data manipulation
from tqdm import tqdm # Pretty progressbars :)
import sys # Np requirement (see setup)
import time # Timy things



#----! SETUP !----#

# Speciffy if you want to rebuilt the data 
# TODO: Replace with commandline argument
REBUILD_DATA = True # Default True

# Specifiy if you want debug prints
# Dissable in production!!!
DEBUG = True 

# Allows for better printing of debug vars
if DEBUG:
    # Only nessisary if debuggin (otherwise there is almost no printing required)
    np.set_printoptions(threshold=sys.maxsize)

class JulianTwanFaceData():

    #----! GENERAL IMAGE SPECS !----#
    # Specify the size of the images (to be imported in)
    IMG_SIZE = 50 # Image size: IMG_SIZExIMG_SIZE [px]
    # Specify if the images need to be importen in grayscale
    USE_GRAYSCALE = True # Default True, best to leave true because shape issues

    #----! FOLDERS, FILES AND LABELS !----#
    # Specify the folders where the (raw) images are stored
    # and specify the output files
    # Also specify the labels 

    # The root directory where all the images are stored in 
    ROOT_DIR = 'julian_twan' # Default: 'julian_twan'
    # Name of the (sub)directory where the images of julian are stored
    JULIAN_DIR = 'julian' 
    # Name of the (sub)directory where the images of twan are stored
    TWAN_DIR = 'twan'

    # The name for the save file
    # File name + the current time to avoid duplication
    save_file_name = f'julian_twan_training_data_{time.time()}.npy' 

    # Give each label a value
    # Julian will be represented with 0 in the training data (instead of the string)
    # and Twan with 1
    # Also link the labels (0,1) to their (raw) source folder
    LABELS = {JULIAN_DIR: 0, TWAN_DIR: 1}

    #----! (TEMP) VARS AND TRAINING_DATA !----#
    # Create variables for counters and other temporary variables
    # and create an array to store the training data in

    # Counters
    # Count (correctly loaded) images of julian
    n_julian = 0 # Start at 0 for correct counting# Start at 0 for correct counting
    # Count (correctly loaded) images of Twan
    n_twan = 0 # Start at 0 for correct counting

    # Training data array/list
    training_data = []

    def make(self):
        ## Creates the training_data from the raw images using the given labels and folders

        #---! SETUP !---#
        # Make a scoped copy of self.ROOT_DIR
        root_dir = self.ROOT_DIR

        #---! MAIN LOOP !---#

        # Loop over all the labels
        # Every labels corresponds with a folder name poining to the directory name 
        # inside the root_dir(=now scoped)
        for label in self.LABELS:
            # Create exact path from root_dir and label(=folder name)
            folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), root_dir, label)
            # If debugging print the folder path
            if DEBUG:
                print(folder_path)

            # Check if the path is actually a dir (preventing unnessesary errors)
            if os.path.isdir(folder_path):
                # Loop over the files in the (label) folder
                # and use pretty progress bars (even if debug != True)
                for img_name in tqdm(os.listdir(folder_path)):
                    # Try/test block to catch errors that might noramlly break
                    # the process because invalid images etc although checks are in place
                    try:
                        # Create the img_path from the file/img name (img_name) and folder_path
                        img_path = os.path.join(folder_path, img_name)
                        # Load image, and convert to grayscale if needed (GRAYSCALE setup var [global])
                        # else load in color (NOT RECOMENDED)
                        img = cv2.imread(img_path, (cv2.IMREAD_GRAYSCALE if self.USE_GRAYSCALE else cv2.IMREAD_COLOR))
                        # Resize image to the specified IMG_SIZE
                        # Creates a square IMG_SIZE by IMG_SIZE
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # Convert the image to a numpy array so it can be stored and accesed by the ai
                        img_arr = np.array(img)
                        # Add to the training data array together with the label (code [0 or 1])
                        # appending this [img_arr, ([0, 1] or [1, 0])] -> [0,1]=Twan [1,0]=Julian
                        self.training_data.append([img_arr, np.eye(2)[self.LABELS[label]]])

                        # Count image
                        if label == self.JULIAN_DIR:
                            self.n_julian += 1
                        elif label == self.TWAN_DIR:
                            self.n_twan += 1
                        else: 
                            if DEBUG:
                                print(f'No valid label? Label={label}')
                        
                    except Exception as e:
                        if DEBUG:
                            print(str(e))
                    
        # Shuffle the training data 
        # This is nessesary because other wise the neural network will have all images of twan together
        # and all images of julian together and develop a non correct bias
        np.random.shuffle(self.training_data)
    
    def save(self):
        ## Saves the training data
        
        # Check if the training data exists and is valid
        if len(self.training_data) >= 1:
            print(f"Saving the training data as: {self.save_file_name}")
            # Save the training data
            np.save(self.save_file_name, self.training_data)
        else:
            # If not give error
            print("ERROR: Training data is empty, run make() first")

    def make_and_save(self):
        # Saves and makes the training data in one go
        # Just for easy use

        print('Making the data...')
        self.make()
        print('Saving the data...')
        self.save()

# If the file is run (NOT IMPORTED)
if __name__ == "__main__":
    # Instanciate the class
    julian_twan_faces = JulianTwanFaceData()
    # Run the save_and_make method to save the data
    julian_twan_faces.make_and_save()