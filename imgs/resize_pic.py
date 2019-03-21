import cv2
import numpy as np

for i in range(1, 6):
    img = cv2.imread('{}.png'.format(i))
    img = cv2.resize(img, (448, 256))
    cv2.imwrite("{}_n.png".format(i), img)
