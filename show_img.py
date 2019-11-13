import cv2
import numpy as np
import matplotlib.pyplot as plt

from process_img import FaceData

FD = FaceData()
training_data = FD.gather_folders_and_labels()
labels = FD.FACES_LABELS
print(training_data[0])
print()
print(labels[np.where(training_data[0][1] == 1.)[0][0]])
plt.imshow(training_data[0][0])
plt.show()