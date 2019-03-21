import numpy as np
import cv2

with open("img_out_1.txt", "r") as file:
    raw_img = file.read()
# raw_img = [10, 12, 1.25]
raw_img = raw_img.split()
raw_img = list(map(lambda x: float(x), raw_img))
raw_img = np.array(raw_img).reshape(2, 720, 1280)


cv2.imshow("channel 0", raw_img[0])
cv2.imshow("channel 1", raw_img[1])
cv2.waitKey(0)


