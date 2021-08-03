import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def IterativeThreshold(img):

    # Convert to array
    img_array = np.array(img).astype(np.float32)
    I = img_array
    theMax = np.max(I)
    theMin = np.min(I)

    # Set initial estimate threshold
    T = (theMax + theMin) / 2

    b = 1
    m, n = I.shape
    while b == 0:
        ifg = 0
        ibg = 0
        fnum = 0
        bnum = 0
        for i in range(1, m):
            for j in range(1, n):
                tmp = I
                if tmp >= T:
                    ifg = ifg + 1
                    fnum = fnum + int(tmp)  # Amount of foreground pixels and the sum of pixel values
                else:
                    ibg = ibg + 1
                    bnum = bnum + int(tmp)  # Amount of background pixels and the sum of pixel values
        # Calculate the average of foreground and background
        foreAvg = int(fnum / ifg)
        backAvg = int(bnum / ibg)
        if T == int((foreAvg + backAvg) / 2):
            b = 0
        else:
            T = int((foreAvg + backAvg) / 2)
    return T


img = cv2.imread("flower.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(gray, (200, 200))  # size
thresh = IterativeThreshold(img)
ret1, th1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
print(ret1)
plt.imshow(th1, cmap=cm.gray)
plt.show()
