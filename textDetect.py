## Imports
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import math
import os
import pytesseract
from PIL import Image

## Loading EAST Detection
#https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

#https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

## get image through cv2


#gets image from path and resizes for EAST detection
def getImg(path):
    aspectRatio = (320,480)
    image = cv2.imread(path)
    h,w,_ = image.shape
    scaledImage = cv2.resize(image, (320,480))
    resizeX, resizeY = w/aspectRatio[0], h/aspectRatio[1]
    return image, scaledImage, resizeX, resizeY


## Takes in image, spits our bounding box (image needs to be w,h multiples of 32)

pathEAST = "frozen_east_text_detection.pb"

#takes in EAST Model path and image
#https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
def applyMaps(path, image):
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    h, w,_ = image.shape
    bgrMean = (123.68, 116.78, 103.94)
    net = cv2.dnn.readNet(path)
    #swapRB swaps R and B so that bgrMean is normal
    blob = cv2.dnn.blobFromImage(image, 1.0, (w,h), bgrMean, 
                                 swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    return scores, geometry

# loop over the number of columns
def boundingBox(scores, geometry):
    boxes = []
    confidences = []
    minConfidence = 0.1
    rows, cols = scores.shape[2:4]
    for y in range(rows):
        for x in range(cols):
            scoreData = scores[0][0][y][x]
            data0 = geometry[0][0][y][x]
            data1 = geometry[0][1][y][x]
            data2 = geometry[0][2][y][x]
            data3 = geometry[0][3][y][x]
            angle = geometry[0][4][y][x]
            if scoreData > minConfidence:
                offsetX, offsetY = x*4, y*4
                h, w = data0 + data2, data1 + data3
                x1 = int(offsetX + (math.cos(angle)*data1) + (math.sin(angle) * data2))
                y1 = int(offsetY - (math.sin(angle)*data1) + (math.cos(angle) * data2))
                x0, y0 = int(x1-w), int(y1-h)
                boxes.append((x0,y0,x1,y1))
                confidences.append(scoreData)
    return boxes, confidences

#destructively resizes boxes to original size
def resizeBoxes(boxes, confidences, resizeX, resizeY):
    boxes = non_max_suppression(np.array(boxes), probs=confidences)
    newBoxes = []
    offSet = 5
    for (x0,y0,x1,y1) in boxes:
        x0 = int(x0 * resizeX) - offSet
        y0 = int(y0 * resizeY) - offSet
        x1 = int(x1 * resizeX) + offSet
        y1 = int(y1 * resizeY) + offSet
        newBox = (x0,y0,x1,y1)
        newBoxes.append(newBox)
    return newBoxes
        
def drawBoxes(image, boxes):
    for (x0,y0,x1,y1) in boxes:
        cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 2)
    return image

path = "testPosters/poster6.jpg"
pathEAST = "frozen_east_text_detection.pb"

def detectText(path, pathEAST):
    image, scaledImage, resizeX, resizeY = getImg(path)
    scores, geometry = applyMaps(pathEAST, scaledImage)
    boxes, confidences = boundingBox(scores, geometry)
    boxes = resizeBoxes(boxes, confidences, resizeX, resizeY)
    return boxes, image
    # image = drawBoxes(image, boxes)
    # cv2.imshow("Detect Text", image)
    # cv2.waitKey(0)
#detectText(path, pathEAST)

def detectVideoText(pathEAST):
    cap = cv2.VideoCapture(0)
    aspectRatio = (320, 480)
    frames = 0
    while True:
        _, frame = cap.read()
        if frames % 10 == 0:
            ## We want to freeze frame and add a loading screen
            h,w,_ = frame.shape
            scaledFrame = cv2.resize(frame, (320,480))
            resizeX, resizeY = w/aspectRatio[0], h/aspectRatio[1]
            scores, geometry = applyMaps(pathEAST, scaledFrame)
            boxes, confidences = boundingBox(scores, geometry)
        newBoxes = resizeBoxes(boxes, confidences, resizeX, resizeY)
        frame = drawBoxes(frame, newBoxes)
        cv2.imshow("Video Detect Text", frame)
        ch = cv2.waitKey(1)
        frames += 1
        if ch & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


## Research for better methods of segmenting text

def binarization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 2)
    restoredCharacters = mendBrokenCharacters(thresh, image)
    dilate = cv2.dilate(thresh, (5,5), iterations = 5)
    #erode = cv2.erode(thresh, (5,5), iterations = 5)
    return dilate

def mendBrokenCharacters(erode, image):
    originalEdge = cv2.Canny(image, 85, 100)
    binaryEdge = cv2.Canny(erode, 85, 100)
    cv2.imshow("BE",binaryEdge)
    cv2.imshow("OE", originalEdge)
    initialEdge = cv2.bitwise_and(originalEdge, binaryEdge)
    cv2.imshow("preious", initialEdge)
    initialEdge = verticleEdgeExtension(initialEdge, originalEdge)
    # for row in range(len(originalEdge)):
    #     newRow = []
    #     for col in range(len(originalEdge[0])):
    #         if originalEdge[row][col] > 0 and binaryEdge[row][col] > 0:
    #             newRow.append((255,255,255))
    #         else:
    #             newRow.append((0,0,0))
    #     initialEdge.append(newRow)
    # print(initialEdge)
    cv2.imshow("Hi", initialEdge)
    pass

def verticleEdgeExtension(edge, oE):
    directions = [(-1,0),(-1,-1),(-1,1)]
    for x in range(len(edge[0])-1):
        for y in range(len(edge)-1):
            if edge[y,x] == 255:
                if edge[y+1,x-1] == edge[y+1,x] == edge[y+1,x+1] == 0:
                    neighbors = (oE[y+1,x-1],oE[y+1,x],oE[y+1,x+1])
                    neighborCount = 3 - neighbors.count(0)
                    if neighborCount == 1:
                        for i in range(len(neighbors)):
                            if neighbors[i] != 0:
                                edge[y+1, x+i-1] += 255
                    # elif neighborCount > 1:
                    #     for i in range(len(neighbors)):
                    #         if neighbors[i] != 0:
                                
    return edge
                                
def horizontalEdgeExtension(edge, oE):
    pass
                                
            

#detectVideoText(pathEAST)
## Takes in bounding box, spits out text
def saveFiles(boxes, image):
    for i in range(len(boxes)):
        x0,y0,x1,y1 = boxes[i]
        box = image[y0:y1, x0:x1]
        betterBox = binarization(box)
        fileName = "tmpFile_"+str(i)+".png"
        cv2.imwrite("tmp/"+fileName, betterBox)

def getText():
    textList = []
    for i in range(len(os.listdir("tmp"))-1):
        fileName = "tmpFile_" + str(i) + ".png"
        text = pytesseract.image_to_string(Image.open("tmp/" + fileName))
        textList.append(text)
    return textList
    
def recognizeText(path, pathEAST):
    boxes, image = detectText(path, pathEAST)
    cv2.imshow("image", image)
    saveFiles(boxes, image)
    textList = getText()
    print(textList)
    cv2.waitKey(0)
    
    
recognizeText(path, pathEAST)