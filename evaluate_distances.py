import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import cv2
import os

#Load embedding and cropped faces
embeddings = np.load('embeddings.npy')
images = np.load('cropped_faces.npy')
print(images.shape)

# Evaluate and print distance between embeddings
A1 = embeddings[0]
A2 = embeddings[1]
A3 = embeddings[2]
A4 = embeddings[3]

B1 = embeddings[4]
B2 = embeddings[5]
B3 = embeddings[6]
B4 = embeddings[7]

for i in range(4):
    for j in range(4):
        dis = distance.euclidean(embeddings[i], embeddings[4+j])
        print('A ' + str(1+i) + ' VS B ' + str(1+j) + ' = ' + str(dis))

print('*' * 55)

for i in range(4):
    for j in range(4):
        dis = distance.euclidean(embeddings[i], embeddings[j])
        print('A ' + str(1+i) + ' VS A ' + str(1+j) + ' = ' + str(dis))

print('*' * 55)


for i in range(4):
    for j in range(4):
        dis = distance.euclidean(embeddings[4+i], embeddings[4+j])
        print('B ' + str(1+i) + ' VS B ' + str(1+j) + ' = ' + str(dis))

# Plot images
plt.figure(1)
i = 1
for img in images:
    plt.subplot(240+i)
    plt.imshow(img)
    if i < 5:
        plt.title('A' + str(i))
    else:
        plt.title('B' + str(i-4))
    i += 1
plt.show()
