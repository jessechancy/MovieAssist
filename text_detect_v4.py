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
import string

## Save Photo Function

def save_photo(img_list, file_name):
    for i in range(len(img_list)):
        img = img_list[i]
        cv2.imwrite("tmp/"+file_name+str(i)+".jpg", img)

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
    for (x0,y0,x1,y1) in boxes:
        x0 = int(x0 * resizeX)
        y0 = int(y0 * resizeY) 
        x1 = int(x1 * resizeX) 
        y1 = int(y1 * resizeY)
        offset_y = (y1-y0)//4
        offset_x = offset_y * 10/7
        x0,y0,x1,y1 = x0 - offset_x, y0 - offset_y, x1 + offset_x, y1 + offset_y
        new_box = (x0,y0,x1,y1)
        new_boxes.append(new_box)
    return new_boxes

def bounding_boxes(path, path_EAST = "frozen_east_text_detection.pb"):
    image, scaled_image, resizeX, resizeY = scaled_img(path)
    scores, geometry = apply_maps(path_EAST, scaled_image)
    boxes, confidences = find_bounds(scores, geometry)
    boxes = resize_boxes(boxes, confidences, resizeX, resizeY)
    return boxes, image

## Preprocessing

## Rearrange and crop bounding boxes to image
def crop_bounding_boxes(bounding_boxes, img):
    img_list = []
    h,w,_ = img.shape
    bounding_boxes = rearrange_box(bounding_boxes)
    #Crop boxes and add to img list
    for (x0,y0,x1,y1) in bounding_boxes:
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        cropped_img = img[y0:y1, x0:x1]
        img_list.append(cropped_img)
    save_photo(img_list, "raw")
    return img_list
    
def rearrange_box(bounding_boxes):
    flagged = False
    min_val = 0
    arranged_boxes = []
    while not flagged:
        avaliable_boxes = list(filter(lambda box: box[1] >= min_val, bounding_boxes))
        if len(avaliable_boxes) != 0:
            #get highest box from remaining boxes
            ref_box = min(avaliable_boxes, key = lambda box: box[1]) #box y0 value
            #find the box middle value
            ref_box_middle = (ref_box[3] - ref_box[1])//2 + ref_box[1]
            #create a list of all boxes on the line with ref_box_middle as middle value
            boxes_on_ref = list(filter(
                                    lambda box: box[1] < ref_box_middle < box[3], 
                                    bounding_boxes
                                           )
                                    )
            #sort the boxes on the line by their x0
            sorted_boxes_line = sorted(boxes_on_ref, key = lambda box: box[1])
            arranged_boxes.extend(sorted_boxes_line)
            #highest y1 value becomes min value for next line
            min_val = max(sorted_boxes_line, key = lambda box: box[3])[3]
        else:
            flagged = True
    return arranged_boxes
    
## Add Filters

#could add image resizing for more consistent processing
def process_img(img):
    h, w, _ = img.shape
    img = resize_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.medianBlur(gray, 5)
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_image = cv2.bitwise_and(thresh1, thresh2)
    processed_image = cv2.resize(processed_image, (w,h))
    return processed_image
    
#process_img helper
def resize_image(img):
    h, w = img.shape[0], img.shape[1]
    image_size_thresh = 100
    if h != 0:
        size_factor = image_size_thresh/h
        img = cv2.resize(img, None, fx = size_factor, fy = size_factor)
    return img

## Check Bounds

def check_bounds(img):
    h, w = img.shape
    img_copy = img.copy()
    rec_boxes = get_contours(img)
    rec_boxes = check_enclosing_rectangles(rec_boxes)
    rec_boxes = remove_edge_rectangles(rec_boxes, w, h)
    if len(rec_boxes) != 0:
        max_height_box = max(rec_boxes, key = lambda x: x[3])
        x0, y0, w, h = max_height_box
        new_img = img_copy[y0:y0+h]
        return new_img
    else:
        return img_copy

#check_bounds helper
def get_contours(img):
    ratio_lim = (0.1, 10)
    img_area = img.shape[0] * img.shape[1]
    area_lim = (15, img_area)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    passed_rec = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w/h
        area = w*h
        if ratio_lim[0] <= aspect_ratio <= ratio_lim[1]:
            if area_lim[0] <= area < area_lim[1]:
                passed_rec.append((x,y,w,h))
    return passed_rec
    
#check_bounds helper
def check_enclosing_rectangles(rec_boxes):
    exterior_boxes = []
    remove_rec_index = set()
    for i in range(len(rec_boxes)):
        rec = rec_boxes[i]
        x, y, w, h = rec
        enclose_set = set()
        for j in range(len(rec_boxes)):
            compare_rec = rec_boxes[j]
            x1, y1, w1, h1 = compare_rec
            if i != j:
                if x < x1 < x1+w1 < x+w and y < y1 < y1+h1 < y+h:
                    remove_rec_index.add(j)
    for i in range(len(rec_boxes)):
        if i not in remove_rec_index:
            exterior_boxes.append(rec_boxes[i])
    return exterior_boxes
    
#check_bounds helper
def remove_edge_rectangles(rec_boxes, width, height):
    non_edge_rec = []
    for (x0,y0,w,h) in rec_boxes:
        if x0 <= 0 or y0 <= 0 or x0 + w >= width or y0 + h >= height:
            continue
        else:
            non_edge_rec.append((x0,y0,w,h))
    return non_edge_rec

## Join Titles

def join_titles(img_list):
    #gets largest box based on area
    max_img, max_height = max_box(img_list)
    #threshold for boxes considered same title
    thresh = max_height//6
    title_list = []
    remainder_list = []
    for img in img_list:
        shape = img.shape
        diff = max_height - shape[0]
        if abs(diff) <= thresh:
            high = diff//2
            low = diff - high
            high_fill = np.zeros((high, shape[1]), dtype = np.uint8)
            low_fill = np.zeros((low, shape[1]), dtype = np.uint8)
            space_width = shape[0]//4 #based on height
            space = np.zeros((max_height, space_width), dtype = np.uint8)
            bg_color = background_color(img)
            if bg_color == 255:
                high_fill[:,:], low_fill[:,:], space[:,:] = 255, 255, 255
            new_img = np.concatenate([high_fill, img, low_fill], axis=0)
            title_list.append(new_img)
            title_list.append(space)
        else:
            remainder_list.append(img)
    concat_title = np.concatenate(title_list, axis=1)
    return [concat_title] + remainder_list
    
def max_box(img_list):
    max_img = None
    max_area = 0
    for img in img_list:
        #area = img.shape[0] * img.shape[1]
        area = img.shape[0]
        if area >= max_area:
            max_area = area
            max_img = img
    return max_img, max_img.shape[0]
    
## Add Borders

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
  
def add_border(img, bg_color):
    border_width = img.shape[0] #image height is border width
    side_fill = np.zeros((border_width, border_width), dtype=np.uint8)
    top_bottom_fill = np.zeros((border_width, img.shape[1]+2*border_width), dtype=np.uint8)
    if bg_color == 255:
        side_fill[:, :] = 255
        top_bottom_fill[:,:] = 255
    img = np.concatenate([side_fill, img, side_fill], axis=1)
    img = np.concatenate([top_bottom_fill, img, top_bottom_fill], axis=0)
    return img
    
## Main Image Processing

#image in cv2 format, numpy array
def pre_processing_main(bounding_boxes, img):
    img_list = crop_bounding_boxes(bounding_boxes, img)
    processed_img_list = []
    for img in img_list:
        processed_image = process_img(img)
        processed_image = check_bounds(processed_image)
        processed_img_list.append(processed_image)
    save_photo(processed_img_list, "processed_bound_checked")
    #join the title letters
    concat_img_list = join_titles(processed_img_list)
    save_photo(concat_img_list, "concat")
    completed_img_list = []
    for img in concat_img_list:
        bg_color = background_color(img)
        bordered_img = add_border(img, bg_color)
        completed_img_list.append(bordered_img)
    save_photo(completed_img_list, "concat")
    return completed_img_list
    
## Main Text Recognition

def text_detect(bounding_boxes, img):
    img_list = pre_processing_main(bounding_boxes, img)
    text_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        img = resize_image(img)
        text = pytesseract.image_to_string(img)
        text_list.append(text)
    return text_list
    
## Spell Check

def spell_check(word_list):
    spell = SpellChecker()
    misspelled = spell.unknown(word_list)
    checked_list = []
    for word in word_list:
        if word == "" or word in string.punctuation:
            continue
        elif " " in word:
            checked_word = " ".join(spell_check(word.split(" ")))
            checked_list.append(checked_word)
        elif word in misspelled:
            corrected_word = spell.correction(word)
            if corrected_word != word:
                checked_list.append(corrected_word)
        else:
            checked_list.append(word)
    return checked_list
    