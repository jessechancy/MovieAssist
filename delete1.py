import cv2
import numpy as np


img = cv2.imread("tmp/tmpFile_14.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
filter = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow("filtered", filter)
threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
threshold = cv2.dilate(threshold, (5,5), iterations = 5)
threshold = cv2.erode(threshold, (5,5), iterations = 5)
cv2.imshow("Thresh before", threshold)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255, 0, 0), -1)
cv2.imshow("img", img)
cv2.imshow("Threshold", threshold)
cv2.imwrite("tmp/tmpFile_15.png", threshold)
cv2.waitKey(0)
