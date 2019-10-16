import numpy as np
import cv2 as cv
import sys

list = [[[0.00057884, 0.00060866]], [[0.00055063, 0.00055412]], [[0.00060251, 0.0006142]], [[0.00061099, 0.00061126]], [[0.00061127,
        0.00059401]], [[0.00056723, 0.00054544]], [[0.00054852, 0.0005222]], [[0.00051996, 0.00050029]], [[0.0004955, 0.00048506]],
        [[0.0004327, 0.00043344]], [[0.00044133, 0.00045759]], [[0.00047369, 0.00050027]], [[0.00051038, 0.0005537]], [[0.00057409,
        0.00061545]], [[0.0006342, 0.00066996]], [[0.00070086, 0.0007652]], [[0.0008221, 0.0008867]], [[0.00095273, 0.00089004]],
        [[0.00082684, 0.00077746]], [[0.00072624, 0.00068827]], [[0.00064738, 0.00060111]], [[0.00059136, 0.00054355]], [[0.00051874,
        0.00051434]], [[0.00048473, 0.0004804]], [[0.00045491, 0.00045303]], [[0.0004464, 0.00042522]], [[0.00047467, 0.00048917]],
        [[0.00050727, 0.00053275]], [[0.00055509, 0.00056515]], [[0.00058112, 0.00059677]], [[0.00062541, 0.0006241]], [[0.00061671,
        0.00061401]]]
ctr = np.array(list).reshape((-1, 1, 2)).astype(np.float)
file = sys.argv[1]
image = cv.imread(file)
cv.drawContours(image, [ctr], -1, (255, 255, 255), 3)
cv.imshow('img', image)
cv.waitKey(0)
