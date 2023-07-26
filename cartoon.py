import numpy as np
import cv2

def read_img(filename):
    img = cv2.imread(filename)
    return img

def edge_detection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur)
    return edges

def color_quantisation(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

img = read_img('./images/lonely.jpg')
line_wdt = 9
blur_value = 7
totalColors = 4

edgeImg = edge_detection(img, line_wdt, blur_value)
img = color_quantisation(img, totalColors)
blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
cartoon = cv2.bitwise_and(blurred, blurred, mask=edgeImg)
cv2.imwrite('Cartoon.jpg', cartoon)
