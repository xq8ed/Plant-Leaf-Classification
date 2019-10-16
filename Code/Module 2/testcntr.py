import cv2 as cv
import numpy as np
import math

test = cv.imread('C://Users//Ganesh//PycharmProjects//ContourDrawing//image.jpg')
test2 = cv.cvtColor(test, cv.COLOR_BGR2GRAY)
test3 = cv.Canny(test2, 30, 200)
list, hierarchy = cv.findContours(test3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
read2 = cv.imread('C://Users//Ganesh//PycharmProjects//ContourDrawing//outputimg.png')
cv.drawContours(read2, list, 0, (0, 0, 0), 3)
cv.imshow('result', read2)
cv.waitKey(0)
cv.destroyAllWindows()
for (i, c) in enumerate(list):
    print("Size of Contour %d: %d" % (i, len(c)))
    size = len(c)
print(list[0])

xtotal = 0
for i in range(0, 2579):
    xtotal = xtotal + list[0][i][0][0]
print("xtotal = %d" % xtotal)
ytotal = 0
for i in range(0, 2579):
    ytotal = ytotal + list[0][i][0][1]
print("ytotal = %d" % ytotal)

# centroid of shape (xbar, ybar)
xbar = xtotal/size
print("xbar = %d" % xbar)
ybar = ytotal/size
print("ybar = %d" % ybar)

# shape descriptor di
di = []
for i in range(0,2579):
    d = math.pow((list[0][i][0][0] - xbar), 2) + math.pow((list[0][i][0][1] - ybar), 2)
#    print(d)
    di = math.sqrt(d)
#    print("Descriptor values = %f" % di)

# di signature
disum = 0
dis = []
for i in range(0, 63):
    disum = disum + di
    dis = di / disum
    print("Final output = %f" % dis)
