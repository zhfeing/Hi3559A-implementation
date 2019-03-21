import numpy as np
import cv2
import matplotlib.pyplot as plt

for i in range(1, 6):

    img = cv2.imread("{}_n.png".format(i))
    print(img.shape)

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    file = open("{}.txt".format(i), "w")

    for c in range(img.shape[2]):
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                file.write("{} ".format(img[h, w, c]))

    file.close()