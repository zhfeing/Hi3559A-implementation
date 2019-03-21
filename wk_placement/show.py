import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 6):

    with open("img_out_{}.txt".format(i), "r") as file:
        raw_img = file.read()
    # raw_img = [10, 12, 1.25]
    raw_img = raw_img.split()
    raw_img = list(map(lambda x: float(x), raw_img))
    raw_img = np.array(raw_img).reshape(2, 256, 448)

    or_img = plt.imread("imgs/{}_n.png".format(i))

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(or_img)
    plt.subplot(3, 1, 2)
    plt.imshow(raw_img[0])
    plt.subplot(3, 1, 3)
    plt.imshow(raw_img[1])
    plt.show()


