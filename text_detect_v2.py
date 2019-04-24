## Aim

"""
Aim: Take in image, return words in image
1. Create bounding boxes for words
2. Preprocessing Techniques
3. Use Pytesseract
4. Spelling Check
"""

## Imports

import cv2
import numpy as np
import os
import pytesseract
from PIL import Image
import math
from imutils.object_detection import non_max_suppression
from spellchecker import SpellChecker

## Bounding Boxes

def scaled_img(path):
    aspect_ratio = (320,480)
    image = cv2.imread(path)
    h,w,_ = image.shape
    scaled_img = cv2.resize(image, (320,480))
    resizeX, resizeY = w/aspect_ratio[0], h/aspect_ratio[1]
    return image, scaled_img, resizeX, resizeY

#takes in EAST Model path and image
#https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
def apply_maps(path, image):
    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    h, w,_ = image.shape
    bgr_mean = (123.68, 116.78, 103.94)
    net = cv2.dnn.readNet(path)
    #swapRB swaps R and B so that bgr mean is normal
    blob = cv2.dnn.blobFromImage(image, 1.0, (w,h), bgr_mean, 
                                 swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)
    return scores, geometry
    
# loop over the number of columns
#https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
def find_bounds(scores, geometry):
    boxes = []
    confidences = []
    min_confidence = 0.1
    rows, cols = scores.shape[2:4]
    for y in range(rows):
        for x in range(cols):
            scoreData = scores[0][0][y][x]
            data0 = geometry[0][0][y][x]
            data1 = geometry[0][1][y][x]
            data2 = geometry[0][2][y][x]
            data3 = geometry[0][3][y][x]
            angle = geometry[0][4][y][x]
            if scoreData > min_confidence:
                offsetX, offsetY = x*4, y*4
                h, w = data0 + data2, data1 + data3
                x1 = int(offsetX + (math.cos(angle)*data1) + (math.sin(angle) * data2))
                y1 = int(offsetY - (math.sin(angle)*data1) + (math.cos(angle) * data2))
                x0, y0 = int(x1-w), int(y1-h)
                boxes.append((x0,y0,x1,y1))
                confidences.append(scoreData)
    return boxes, confidences

#destructively resizes boxes to original size
#https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
def resize_boxes(boxes, confidences, resizeX, resizeY):
    boxes = non_max_suppression(np.array(boxes), probs=confidences)
    new_boxes = []
    offSet = 5
    for (x0,y0,x1,y1) in boxes:
        x0 = int(x0 * resizeX) - offSet
        y0 = int(y0 * resizeY) - offSet
        x1 = int(x1 * resizeX) + offSet
        y1 = int(y1 * resizeY) + offSet
        new_box = (x0,y0,x1,y1)
        new_boxes.append(new_box)
    return new_boxes

def bounding_boxes(path, path_EAST = "frozen_east_text_detection.pb"):
    image, scaled_image, resizeX, resizeY = scaled_img(path)
    scores, geometry = apply_maps(path_EAST, scaled_image)
    boxes, confidences = find_bounds(scores, geometry)
    boxes = resize_boxes(boxes, confidences, resizeX, resizeY)
    return boxes, image

## Image Preprocessing

#Crops image based on given bounding boxes and return a list of cropped images
def crop_bounding_box(bounding_boxes, img):
    img_list = []
    for (x0,y0,x1,y1) in bounding_boxes:
        cropped_img = img[y0:y1, x0:x1]
        img_list.append(cropped_img)
    return img_list

def pre_processing_main(bounding_boxes, img):
    img_list = crop_bounding_box(bounding_boxes, img)
    for i in range(len(img_list)):
        cv2.imwrite("bounding_boxes/" + str(i)+ ".jpg", img_list[i])
    processed_img_list = []
    for img in img_list:
        #img = cv2.imread("bounding_boxes/"+str(i)+".png")
        processed_img = process_image(img)
        processed_img = check_bound(processed_img)
        processed_img_list.append(processed_img)
    for i in range(len(processed_img_list)):
        cv2.imwrite("processed_boxes/" + str(i)+ ".jpg", processed_img_list[i])
    concat_img_list = join_titles(processed_img_list)
    completed_img_list = []
    for img in concat_img_list:
        bg_color = background_color(img)
        bordered_img = add_border(img, bg_color)
        completed_img_list.append(bordered_img)
    for i in range(len(completed_img_list)):
        cv2.imwrite("processed_boxes/final" + str(i)+ ".jpg", completed_img_list[i])
    return completed_img_list

def add_border(img, bg_color):
    border_width = img.shape[0]
    side_fill = np.zeros((border_width, border_width), dtype = np.uint8)
    top_bottom_fill = np.zeros((border_width, img.shape[1]+2*border_width), dtype = np.uint8)
    if bg_color == 255:
        side_fill[:,:] = 255
        top_bottom_fill[:,:] = 255
    img = np.concatenate([side_fill, img, side_fill], axis = 1)
    img = np.concatenate([top_bottom_fill, img, top_bottom_fill], axis = 0)
    return img

def join_titles(img_list):
    max_img, max_val = max_box(img_list)
    print(max_img, len(img_list))
    threshold = max_val//4
    title_list = []
    remainder_list = []
    for img in img_list:
        shape = img.shape
        print(img.shape)
        diff = max_val - shape[0]
        if abs(diff) <= threshold:
            high = diff//2
            low = diff - high
            high_fill = np.zeros((high, shape[1]), dtype = np.uint8)
            low_fill = np.zeros((low, shape[1]), dtype = np.uint8)
            space = np.zeros((max_val, 100), dtype = np.uint8)
            bg_color = background_color(img)
            print(bg_color, "bg_color")
            if bg_color == 255:
                high_fill[:,:] = 255
                low_fill[:,:] = 255
                space[:,:] = 255
            new_img = np.concatenate([high_fill, img, low_fill], axis = 0)
            print(new_img)
            title_list.append(new_img)
            title_list.append(space)
        else:
            remainder_list.append(img)
    concat_title = np.concatenate(title_list, axis = 1)
    cv2.imwrite("processed_boxes/" + "hi"+ ".jpg", concat_title)
    return [concat_title] + remainder_list

# Dictionary zipping technique taken from https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
def background_color(img):
    threshold = 1 #how many pixels to check
    color, left_count = np.unique(img[:, :1], return_counts = True)
    left = dict(zip(color,left_count))
    
    color, right_count = np.unique(img[:, -1:], return_counts = True)
    right = dict(zip(color,right_count))
    
    color, top_count = np.unique(img[:1, :], return_counts = True)
    top = dict(zip(color,top_count))
    
    color, bottom_count = np.unique(img[-1:, :], return_counts = True)
    bottom = dict(zip(color,bottom_count))
    
    black_pixels = 0
    white_pixels = 0
    for side in [left, right, top, bottom]:
        if 0 in side:
            black_pixels += side[0]
        if 255 in side:
            white_pixels += side[255]
    if white_pixels >= black_pixels:
        return 255
    else:
        return 0
    
def max_box(img_list):
    max_img = None
    max_val = 0
    for img in img_list:
        if img.shape[0] >= max_val:
            max_val = img.shape[0]
            max_img = img
    return max_img, max_val
    

def check_bound(img):
    def check_top(img, limit):
        flagged = False
        while not flagged:
            limit -= 1
            if limit > 0:
                if len(np.unique(img[0])) == 1:
                    flagged = True
                else:
                    img = np.delete(img, 0, 0)
            else:
                print("hit limit top")
                flagged = True
        return img
    def check_bottom(img, limit):
        flagged = False
        while not flagged:
            limit -= 1
            if limit > 0:
                if len(np.unique(img[len(img)-1])) == 1:
                    flagged = True
                else:
                    img = np.delete(img, len(img)-1, 0)
            else:
                print("hit limit bottom")
                flagged = True
        return img
    def check_left(img, limit):
        flagged = False
        while not flagged:
            limit -= 1
            if limit > 0:
                if len(np.unique(img[:,0])) == 1:
                    flagged = True
                else:
                    img = np.delete(img, 0, 1)
            else:
                print("hit limit left")
                flagged = True
        return img
    def check_right(img, limit):
        flagged = False
        while not flagged:
            limit -= 1
            if limit > 0:
                if len(np.unique(img[:,len(img[0])-1])) == 1:
                    flagged = True
                else:
                    img = np.delete(img, len(img[0])-1, 1)
            else:
                print("hit limit right")
                flagged = True
        return img
    # img = check_top(img, 10)
    # img = check_bottom(img, 10)
    # img = check_left(img, 10)
    # img = check_right(img, 10)
    return img
            
def process_image(img):
    print(img.shape, "test")
    h,w,_ = img.shape
    # Resizes to allow for better processing
    img = resize_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pre_blur_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.medianBlur(gray,5)
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_and(thresh1, thresh2)
    processed_image = cv2.bitwise_or(pre_blur_thresh, thresh)
    processed_image = cv2.resize(processed_image, (w,h))
    return processed_image

def resize_image(img):
    h,w = img.shape[0], img.shape[1]
    image_size_threshold = 100
    if h != 0:
        size_factor = image_size_threshold/h
        img = cv2.resize(img, None, fx = size_factor, fy = size_factor)
    return img
    
## Main Text Detection

def text_detect(bounding_boxes, img):
    img_list = pre_processing_main(bounding_boxes,img)
    text_list = []
    for img in img_list:
        img = resize_image(img)
        text = pytesseract.image_to_string(img)
        text_list.append(text)
    print(text_list)
    return text_list

# img = cv2.imread("processed_boxes/final0.png")
# h,w,_ = img.shape
# image_size_threshold = 100
# size_factor = image_size_threshold/h
# img = cv2.resize(img, None, fx = size_factor, fy = size_factor)
# text = pytesseract.image_to_string(img)
# print(text)
# img = cv2.imread("bounding_boxes/0.jpg")
# h,w,_ = img.shape
# w = w//4
# h = h//4
# img = cv2.resize(img, (w,h))
# processed_image = process_image(img)
# cv2.imwrite("bounding_boxes/filter_test.jpg", processed_image)

## Spelling Check

def spell_check(word_list):
    print(word_list)
    spell = SpellChecker()
    misspelled = spell.unknown(word_list)
    print(misspelled)
    checked_list = []
    for word in word_list:
        if " " in word:
            checked_word = " ".join(spell_check(word.split(" ")))
            checked_list.append(checked_word)
        elif word in misspelled:
            corrected_word = spell.correction(word)
            if corrected_word != word:
                checked_list.append(corrected_word)
        else:
            checked_list.append(word)
    return checked_list