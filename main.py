import math

import cv2
import matplotlib
import numpy
import numpy as np

img = cv2.imread('road.jpg', cv2.IMREAD_COLOR)

scale_percent = 40  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('resized',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#hsv
hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]


blur = cv2.GaussianBlur(v, (3, 3), 0)
cv2.imshow("blur"+str(7), blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
import scipy
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # generate some sample data
#
#
#
#
# # create the x and y coordinate arrays (here we just use pixel indices)
# xx, yy = np.mgrid[0:v.shape[0], 0:v.shape[1]]
#
# # create the figure
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, v ,rstride=1, cstride=1, cmap=plt.cm.gray,
#         linewidth=0)
#
# # show it
# plt.show()

dst = cv2.Canny(v,100,200)
cv2.imshow('edges',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)


cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

cv2.waitKey()
