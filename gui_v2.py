## Aim
"""
Run this file
Main Graphical User Interface
"""

## Imports

from tkinter import *
import cv2
import string
from PIL import Image, ImageTk
from text_detect_v2 import text_detect, bounding_boxes, spell_check
from movie_processor_v2 import movie_info, movie_poster, movie_recommend

## Notes

"""
Coordinates are always given in (x0,y0,x1,y1) format
Modes: Login, Edit, Homescreen, Recommednation
"""

## Initialize Data

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
    
def init(data):
    data.mode = "login" #starts off at the login page
    #center
    data.cx, data.cy = data.width//2, data.height//2
    #buttons
    data.LOGIN_BUTTON = {"Login": size_to_canvas(data, (40,60,60,75))}
    data.MAINSCREEN_BUTTON = {"Take Photo": size_to_canvas(data,(10,10,20,20)),
                              "Search": size_to_canvas(data,(80,10,90,20))}
    data.EDIT_BUTTON = {"Done": size_to_canvas(data, (10,10,20,20))}
    #inputs
    data.username = ""
    data.movie_title = ""
    #for editing
    data.bounding_boxes = None
    data.clicked_boxes = []
    data.orig_image = None

## General

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

def redraw_all(canvas, data):
    if data.mode == "login": login_redraw_all(canvas, data)
    elif data.mode == "edit": edit_redraw_all(canvas, data)
    elif data.mode == "homescreen": homescreen_redraw_all(canvas, data)
    elif data.mode == "recommendations": recommendations_redraw_all(canvas, data)

def timer_fired(data):
    pass
    
## Helper Functions

def in_button(x,y,button_coord):
    (x0,y0,x1,y1) = button_coord
    if x0 <= x <= x1:
        if y0 <= y <= y1:
            return True
    return False
    
## Login (Goes to Homescreen)

def login_mouse_pressed(event, data):
    if in_button(event.x, event.y, data.LOGIN_BUTTON["Login"]):
        data.mode = "homescreen"

def login_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Purple")
    x0,y0,x1,y1 = data.LOGIN_BUTTON["Login"]
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
    elif event.keysym in string.lowercase:
        data.username += event.keysym

## Homescreen (Go to Recommendations or Edit)

#rf is resize factor
def get_bounding_box(data):
    boxes, img = bounding_boxes("test_posters/poster5.jpg")
    data.orig_img = img
    img, data.shrink_factor = resize_img(img, data.width*2//3)
    boxes, data.offsetX, data.offsetY = resize_box(img, boxes, data.shrink_factor,
                                                   data.cx, data.cy)
    return img, boxes
    
def resize_box(img, boxes, shrink_factor, cx, cy):
    offsetX = cx - img.width()//2
    offsetY = cy - img.height()//2
    resized_boxes = []
    for (x0,y0,x1,y1) in boxes:
        x0,y0,x1,y1 = x0*shrink_factor,y0*shrink_factor,x1*shrink_factor,y1*shrink_factor
        x0,y0,x1,y1 = x0 + offsetX, y0 + offsetY, x1 + offsetX, y1 + offsetY
        resized_boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return resized_boxes, offsetX, offsetY
    
#img taken in is in cv2 format, want to change to array then tkinter format
def resize_img(img, resize_height):
    #reformat image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    w,h = img.size
    height = resize_height
    width = (w/h) * resize_height
    #ANTIALIAS preserves image quality
    img = img.resize((int(width),int(height)), Image.ANTIALIAS)
    tkinter_img = ImageTk.PhotoImage(img)
    shrink_factor = resize_height/h
    return tkinter_img, shrink_factor
    
def homescreen_mouse_pressed(event, data):
    if in_button(event.x, event.y, data.MAINSCREEN_BUTTON["Take Photo"]):
        data.img, data.bounding_boxes = get_bounding_box(data)
        data.mode = "edit"
    elif in_button(event.x, event.y, data.MAINSCREEN_BUTTON["Search"]):
        #movie_info returns False if not found
        if movie_info(data.movie_title) != False:
            _, data.input_movie = movie_info(data.movie_title)
            #pick the first search result to recommend
            recommended_movies_id = movie_recommend(data.input_movie)
            #pick the first recommended movie to display
            _, data.recommended_movie = movie_info(recommended_movies_id[0])
            poster = movie_poster(data.input_movie['poster_path'])
            poster = resize_poster(data,poster)
            data.poster = ImageTk.PhotoImage(poster)
            rec_poster = movie_poster(data.recommended_movie["poster_path"])
            rec_poster = resize_poster(data, rec_poster)
            data.recommended_poster = ImageTk.PhotoImage(rec_poster)
            data.mode = "recommendations"

def resize_poster(data,poster):
    height = data.height*2/3
    w, h = poster.size
    width = w * height/h
    poster = poster.resize((int(width),int(height)))
    return poster

def homescreen_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Green")
    #draw buttons
    for button_name in data.MAINSCREEN_BUTTON:
        (x0,y0,x1,y1) = data.MAINSCREEN_BUTTON[button_name]
        canvas.create_rectangle(x0,y0,x1,y1, fill = "Blue")
        canvas.create_text((x1-x0)//2 + x0, (y1-y0)//2 + y0, anchor = "c", text = button_name)
    #draw input
    canvas.create_line(data.cx//2, data.cy, data.cx*3//2, data.cy, fill = "Blue")
    canvas.create_text(data.cx, data.cy-10, anchor = "s", text = data.movie_title, font="Arial 50 bold")
    
def homescreen_key_pressed(event, data):
    if event.keysym == "BackSpace":
        if len(data.movie_title) > 0:
            data.movie_title = data.movie_title[:-1]
    elif event.char in string.printable and event.keysym != "BackSpace":
        data.movie_title += event.char

## Edit (Goes to Homescreen)

def is_clicked(x,y,x0,y0,x1,y1,box_width=5):
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

#undo the resize box above
def undo_resize_box(data, box_coord):
    (x0,y0,x1,y1) = box_coord
    x0,y0,x1,y1 = x0-data.offsetX, y0-data.offsetY, x1-data.offsetX, y1-data.offsetY
    x0,y0,x1,y1 = (x0/data.shrink_factor, y0/data.shrink_factor, 
                   x1/data.shrink_factor, y1/data.shrink_factor)
    return (int(x0),int(y0),int(x1),int(y1))

def edit_mouse_pressed(event, data):
    if in_button(event.x, event.y, data.EDIT_BUTTON["Done"]):
        #undo the shrinking of bounding boxes
        reg_size_boxes = []
        for box in data.bounding_boxes:
            new_box = undo_resize_box(data, box)
            reg_size_boxes.append(new_box)
        #using text recognition
        data.text = text_detect(reg_size_boxes, data.orig_img)
        data.text = spell_check(data.text)
        data.movie_title = data.text[0]
        data.mode = "homescreen"
    #holds clicked side or False value for every bounding box
    data.clicked_boxes = []
    for (x0,y0,x1,y1) in data.bounding_boxes:
        clicked_side = is_clicked(event.x,event.y,x0,y0,x1,y1)
        data.clicked_boxes.append(clicked_side)

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

def edit_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Yellow")
    canvas.create_image(data.cx, data.cy, anchor = "c", image = data.img)
    for (x0,y0,x1,y1) in data.bounding_boxes:
        canvas.create_rectangle(x0,y0,x1,y1,width=3)
    for button_name in data.EDIT_BUTTON:
        x0,y0,x1,y1 = data.EDIT_BUTTON[button_name]
        canvas.create_rectangle(x0,y0,x1,y1, fill = "Purple")
        canvas.create_text((x1-x0)//2 + x0, (y1-y0)//2 + y0, text = button_name)

## Recommendations

def recommendations_mouse_pressed(event, data):
    pass

def recommendations_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height)
    canvas.create_image(data.cx//2, data.cy, anchor = "c", image = data.poster)
    canvas.create_image(data.cx + data.cx//2, data.cy, anchor = "c", image = data.recommended_poster)

## Run Function

# Taken from the 15-112 webpage
def run(width=1000, height=600):
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