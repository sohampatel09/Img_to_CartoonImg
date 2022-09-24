import numpy as np
import cv2

def read_img(filename):
    img = cv2.imread(filename)
    return img

def edge_detection(img, line_wdth, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, line_wdth, blur)
    return edges

def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    rect, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

img = read_img("img5.jpeg")
line_wdth = 13
blur_value = 7
totalColors = 10

edgeImg = edge_detection(img, line_wdth, blur_value)
img = color_quantization(img, totalColors)
blurred = cv2.bilateralFilter(img, d=9, sigmaColor=7, sigmaSpace=7)
cartoon = cv2.bitwise_and(blurred, blurred, mask = edgeImg)
cv2.imwrite("cartoon2.jpg", cartoon)
