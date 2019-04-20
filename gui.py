# Updated Animation Starter Code

import cv2
from tkinter import *
import string
from PIL import Image, ImageTk
from text_detect_v2 import text_detect, bounding_boxes

## Notes

#Coordinates are always given in (x0,y0,x1,y1) format

####################################
# customize these functions
###################################
#Modes: Login, Edits, Homescreen, Recommendations, 

def init(data):
    data.mode = "login"
    data.cx = data.width//2
    data.cy = data.height//2
    button_size = data.width//10
    data.LOGIN_BUTTON = (data.cx-button_size, data.cy+button_size, 
                       data.cx+button_size, data.cy+button_size*2)
    data.MAINSCREEN_BUTTON = [(size_to_canvas(data,(10,10,20,20)), "Take Photo"),
                              (size_to_canvas(data,(80,10,90,20)), "Search")]
    data.EDIT_BUTTON = [(size_to_canvas(data, (10,10,20,20)), "Done")]
    data.username = ""
    data.movie_title = ""
    data.bounding_boxes = None
    data.clicked_boxes = []
    
def mouse_pressed(event, data):
    if data.mode == "login": login_mouse_pressed(event, data)
    elif data.mode == "edit": edit_mouse_pressed(event, data)
    elif data.mode == "homescreen": homescreen_mouse_pressed(event, data)
    elif data.mode == "recommendations": recommendations_mouse_pressed(event, data)

def key_pressed(event, data):
    if data.mode == "login": login_key_pressed(event, data)
    elif data.mode == "edit": edit_key_pressed(event, data)
    elif data.mode == "homescreen": homescreen_key_pressed(event, data)
    elif data.mode == "recommendations": recommendations_key_pressed(event, data)
    pass

def timer_fired(data):
    pass

def redraw_all(canvas, data):
    if data.mode == "login": login_redraw_all(canvas, data)
    elif data.mode == "edit": edit_redraw_all(canvas, data)
    elif data.mode == "homescreen": homescreen_redraw_all(canvas, data)
    elif data.mode == "recommendations": recommendations_redraw_all(canvas, data)

## Helper Functions

def in_button(x,y,button_coord):
    (x0,y0,x1,y1) = button_coord
    if x0 <= x <= x1:
        if y0 <= y <= y1:
            return True
    return False

#input a numerical value out of 100 and it would size to canvas
def size_to_canvas(data, button_coord):
    x0,y0,x1,y1 = button_coord
    width_scale = data.width/100
    height_scale = data.height/100
    x0 = x0 * width_scale
    y0 = y0 * height_scale
    x1 = x1 * width_scale
    y1 = y1 * height_scale
    return (x0,y0,x1,y1)

##

class Movie(object):
    pass

## Login Page

def login_mouse_pressed(event, data):
    if in_button(event.x, event.y, data.LOGIN_BUTTON):
        data.mode = "homescreen"

def login_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Purple")
    x0,y0,x1,y1 = data.LOGIN_BUTTON
    canvas.create_rectangle(x0,y0,x1,y1,fill="Green")
    canvas.create_text((x0+x1)//2, (y0+y1)//2, text="Login",
                       font="Arial 40 bold")
    canvas.create_line(data.cx//2, data.cy, data.cx*3//2, data.cy, fill = "Blue")
    canvas.create_text(data.cx, data.cy-10, anchor = "s", text = data.username, font="Arial 50 bold")

def login_key_pressed(event, data):
    print(event.keysym)
    if event.keysym == "BackSpace":
        if len(data.username) > 0:
            data.username = data.username[:-1]
    elif event.keysym in string.printable and event.keysym != "BackSpace":
        data.username += event.keysym

## Homescreen

def get_bounding_box(data):
    data.rf = 5
    boxes, img = bounding_boxes("test_posters/poster8.jpg")
    data.orig_img = img
    img,w,h = resize_img(img, data.rf)
    data.offsetX = data.cx - w//2
    data.offsetY = data.cy - h//2
    resized_boxes = []
    for box in boxes:
        x0,y0,x1,y1 = box
        x0,y0,x1,y1 = x0//data.rf,y0//data.rf,x1//data.rf,y1//data.rf
        x0,y0,x1,y1 = x0 + data.offsetX, y0 + data.offsetY, x1 + data.offsetX, y1 + data.offsetY
        resized_boxes.append((x0,y0,x1,y1))
    return resized_boxes, img

#Image resize line taken from https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
def resize_img(img, resize_factor):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    resize_height = img.size[1] // resize_factor
    resize_width = img.size[0] // resize_factor
    img = img.resize((resize_width,resize_height), Image.ANTIALIAS)
    tkinter_img = ImageTk.PhotoImage(img)
    return tkinter_img, resize_width, resize_height

def homescreen_mouse_pressed(event, data):
    if in_button(event.x, event.y, data.MAINSCREEN_BUTTON[0][0]):
        data.bounding_boxes, data.img = get_bounding_box(data)
        data.mode = "edit"
    elif in_button(event.x, event.y, data.MAINSCREEN_BUTTON[1][0]):
        data.mode = "recommendations"

def homescreen_redraw_all(canvas, data):
    #Search, Import Photo, #
    canvas.create_rectangle(0,0,data.width,data.height,fill="Green")
    for button in data.MAINSCREEN_BUTTON:
        (x0,y0,x1,y1), name = button
        canvas.create_rectangle(x0,y0,x1,y1, fill = "Blue")
        canvas.create_text((x1-x0)//2 + x0, (y1-y0)//2 + y0, anchor = "c", text = name)
    canvas.create_line(data.cx//2, data.cy, data.cx*3//2, data.cy, fill = "Blue")
    canvas.create_text(data.cx, data.cy-10, anchor = "s", text = data.movie_title, font="Arial 50 bold")

def homescreen_key_pressed(event, data):
    if event.keysym == "BackSpace":
        if len(data.movie_title) > 0:
            data.movie_title = data.movie_title[:-1]
    elif event.char in string.printable and event.keysym != "BackSpace":
        data.movie_title += event.char

## Edits

def is_clicked(x,y,x0,y0,x1,y1):
    box_width = 5
    #left
    if abs(x-x0) <= box_width and y0 < y < y1:
        return "left"
    #right
    elif abs(x-x1) <= box_width and y0 < y < y1:
        return "right"
    #top
    elif abs(y-y0) <= box_width and x0 < x < x1:
        return "top"
    #bottom
    elif abs(y-y1) <= box_width and x0 < x < x1:
        return "bottom"
    return False
    
def reformat_box(data, box_coord):
    (x0,y0,x1,y1) = box_coord
    x0,y0,x1,y1 = x0-data.offsetX, y0-data.offsetY, x1-data.offsetX, y1-data.offsetY
    x0,y0,x1,y1 = x0*data.rf, y0*data.rf, x1*data.rf, y1*data.rf
    return (x0,y0,x1,y1)
    
def edit_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Yellow")
    canvas.create_image(data.cx, data.cy, anchor = "c", image = data.img)
    for (x0,y0,x1,y1) in data.bounding_boxes:
        canvas.create_rectangle(x0,y0,x1,y1,width=3)
    for ((x0,y0,x1,y1),name) in data.EDIT_BUTTON:
        canvas.create_rectangle(x0,y0,x1,y1, fill = "Purple")
        canvas.create_text((x1-x0)//2 + x0, (y1-y0)//2 + y0, text = name)

def edit_mouse_pressed(event,data):
    data.clicked_boxes = []
    for (x0,y0,x1,y1) in data.bounding_boxes:
        clicked = is_clicked(event.x,event.y,x0,y0,x1,y1)
        data.clicked_boxes.append(clicked)
    if in_button(event.x, event.y, data.EDIT_BUTTON[0][0]):
        tmp_bounding_boxes = []
        for box in data.bounding_boxes:
            new_box = reformat_box(data, box)
            tmp_bounding_boxes.append(new_box)
        data.bounding_boxes = tmp_bounding_boxes
        data.text = text_detect(data.bounding_boxes, data.orig_img)
        data.movie_title = data.text[0]
        data.mode = "homescreen"
        

def edit_key_pressed(event, data):
    shift = 2
    for side in ["left", "right", "top", "bottom"]:
        if side in data.clicked_boxes:
            i = data.clicked_boxes.index(side)
            x0,y0,x1,y1 = data.bounding_boxes[i]
            if side == "left":
                if event.keysym == "Right":
                    x0 += shift
                elif event.keysym == "Left":
                    x0 -= shift
            elif side == "right":
                if event.keysym == "Right":
                    x1 += shift
                elif event.keysym == "Left":
                    x1 -= shift
            elif side == "top":
                if event.keysym == "Up":
                    y0 -= shift
                elif event.keysym == "Down":
                    y0 += shift
            elif side == "bottom":
                if event.keysym == "Up":
                    y1 -= shift
                elif event.keysym == "Down":
                    y1 += shift
            data.bounding_boxes[i] = (x0,y0,x1,y1)
    
## Recommendations

def recommendations_mouse_pressed(event, data):
    pass

def recommendations_redraw_all(canvas, data):
    pass
    
####################################
# use the run function as-is
####################################

# Taken from the 15-112 webpage
def run(width=300, height=300):
    def redraw_all_wrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redraw_all(canvas, data)
        canvas.update()    

    def mouse_pressed_wrapper(event, canvas, data):
        mouse_pressed(event, data)
        redraw_all_wrapper(canvas, data)

    def key_pressed_wrapper(event, canvas, data):
        key_pressed(event, data)
        redraw_all_wrapper(canvas, data)

    def timer_fired_wrapper(canvas, data):
        timer_fired(data)
        redraw_all_wrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timer_delay, timer_fired_wrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timer_delay = 100 # milliseconds
    root = Tk()
    #creates Entry widget
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mouse_pressed_wrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            key_pressed_wrapper(event, canvas, data))
    timer_fired_wrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(1000, 600)