import cv2
import numpy as np
import matplotlib.pyplot as plt

import random

### OLD TEST
# from process_img import FaceData
# FD = FaceData()
# training_data = FD.gather_folders_and_labels()
# labels = FD.FACES_LABELS
# print(training_data[0])
# print()
# print(labels[np.where(training_data[0][1] == 1.)[0][0]])
# plt.imshow(training_data[0][0])
# plt.show()


file_data = np.load('training_data.npy', allow_pickle=True)
training_data = file_data[0]
randon_nr = random.randint(0, (len(training_data)-1))
labels = file_data[1]

print(training_data[randon_nr])
print()
print(labels[np.where(training_data[randon_nr][1] == 1.)[0][0]])
plt.imshow(training_data[randon_nr][0])
plt.show()
