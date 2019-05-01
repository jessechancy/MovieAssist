## Aim
"""
Run this file
Main Graphical User Interface
"""

## Imports

from tkinter import *
from tkinter import filedialog
import cv2
import string
from PIL import Image, ImageTk
from text_detect_v3 import text_detect, bounding_boxes, spell_check
from movie_processor_v2 import movie_info, movie_poster, movie_recommend, display_info
import numpy as np
import threading
import pandas as pd
import time

## Objects

class Movie_Poster(object):
    def __init__(self, coord, poster_path=None, title=None):
        self.title = title
        self.x, self.y = coord
        if poster_path != None:
            self.poster_img = movie_poster(poster_path)
    def draw_poster(self, canvas):
        self.poster = ImageTk.PhotoImage(self.poster_img)
        canvas.create_image(self.x, self.y, anchor="c", image=self.poster)
    def resize(self, resize_height):
        self.resize_height = resize_height
        w, h = self.poster_img.size
        height = resize_height
        width = w/h * height
        self.poster_img = self.poster_img.resize((int(width),int(height)), Image.ANTIALIAS)
        return height/h #shrink factor
    def is_clicked(self, click_x, click_y):
        w, h = self.poster_img.size
        x0, y0, x1, y1 = self.x-w//2, self.y-h//2, self.x+w//2, self.y+h//2
        if x0 <= click_x <= x1 and y0 <= click_y <= y1:
            return True
        return False

#resize before getting bounding boxes
class Movie_Edit(Movie_Poster):
    def __init__(self, coord):
        super().__init__(coord)
        poster = filedialog.askopenfilename()
        self.boxes, poster = bounding_boxes(poster)
        self.orig_poster = poster
        poster = cv2.cvtColor(poster, cv2.COLOR_BGR2RGB)
        self.poster_img = Image.fromarray(poster)
        self.w, self.h = self.poster_img.size
        self.movie_name = None
        self.bounding_boxes = []
    def get_bounding_boxes(self):
        w, h = self.w, self.h
        for box in self.boxes:
            object_box = Bounding_Box(box)
            print(box, self.resize_height, h, object_box.get_coord(), "infoj")
            object_box.resize(w, h, self.resize_height, self.x, self.y)
            print(box, self.resize_height, h, object_box.get_coord(), "after")
            self.bounding_boxes.append(object_box)
    def check_spelling(self):
        self.movie_name = spell_check(self.movie_name)[0]
        return self.movie_name
    def find_text(self):
        box_coord = []
        for box in self.bounding_boxes:
            box.undo_resize()
            box_coord.append(box.get_coord())
        self.movie_name = text_detect(box_coord, self.orig_poster)
        print(self.movie_name)
        return self.movie_name
    def draw_poster(self, canvas):
        super().draw_poster(canvas)
        for box in self.bounding_boxes:
            box.draw_box(canvas)
        
class Bounding_Box(object):
    def __init__(self, coord):
        self.x0,self.y0,self.x1,self.y1 = coord
        self.x0_og,self.y0_og,self.x1_og,self.y1_og = coord
        self.clicked_side = None
        self.click_w = 5
    def draw_box(self, canvas):
        canvas.create_rectangle(self.x0, self.y0, self.x1, self.y1, width=3, outline="Green")
    def is_clicked(self, x, y):
        if abs(x-self.x0) <= self.click_w and self.y0 < y < self.y1:
            self.clicked_side = "left"
        elif abs(x-self.x1) <= self.click_w and self.y0 < y < self.y1:
            self.clicked_side = "right"
        elif abs(y-self.y0) <= self.click_w and self.x0 < x < self.x1:
            self.clicked_side = "top"
        elif abs(y-self.y1) <= self.click_w and self.x0 < x < self.x1:
            self.clicked_side = "bottom"
        else:
            self.clicked_side = None
    def shift_side(self, keysym):
        shift = 2
        if self.clicked_side == "left":
            if keysym == "Right":
                self.x0 += shift
            elif keysym == "Left":
                self.x0 -= shift
        elif self.clicked_side == "right":
            if keysym == "Right":
                self.x1 += shift
            elif keysym == "Left":
                self.x1 -= shift
        elif self.clicked_side == "top":
            if keysym == "Up":
                self.y0 -= shift
            elif keysym == "Down":
                self.y0 += shift
        elif self.clicked_side == "bottom":
            if keysym == "Up":
                self.y1 -= shift
            elif keysym == "Down":
                self.y1 += shift
    def get_coord(self):
        return (self.x0, self.y0, self.x1, self.y1)
    #resized height is after resizing image height, h is image original height
    def resize(self, w, h, resize_height, cx, cy):
        self.shrink_factor = resize_height/h
        resize_width = w * self.shrink_factor
        self.offset_x = cx - resize_width//2
        self.offset_y = cy - resize_height//2
        self.x0 = int(self.x0 * self.shrink_factor)
        self.y0 = int(self.y0 * self.shrink_factor)
        self.x1 = int(self.x1 * self.shrink_factor)
        self.y1 = int(self.y1 * self.shrink_factor)
        self.x0 += self.offset_x
        self.y0 += self.offset_y
        self.x1 += self.offset_x
        self.y1 += self.offset_y
    def undo_resize(self):
        self.x0 -= self.offset_x
        self.y0 -= self.offset_y
        self.x1 -= self.offset_x
        self.y1 -= self.offset_y
        self.x0 = int(self.x0/self.shrink_factor)
        self.y0 = int(self.y0/self.shrink_factor)
        self.x1 = int(self.x1/self.shrink_factor)
        self.y1 = int(self.y1/self.shrink_factor)
        

class Button(object):
    def __init__(self, name, coord, font_size, color="White"):
        self.name = name
        self.x0, self.y0, self.x1, self.y1 = coord
        self.font_size = font_size
        self.color = color
    def draw_button(self, canvas):
        canvas.create_text((self.x0+self.x1)//2, (self.y0+self.y1)//2, 
                           text=self.name, font="Arial "+str(self.font_size)+" bold",
                           fill=self.color)
        canvas.create_line(self.x0, self.y1-5, self.x1, self.y1-5,fill="#E50914",
                           width=3)
    def is_clicked(self, click_x, click_y):
        if self.x0 <= click_x <= self.x1 and self.y0 <= click_y <= self.y1:
            return True
        return False

class Rating_Star(object):
    #Bool for full_star On Or Off
    def __init__(self, coord, size, full_star):
        self.x, self.y = coord
        self.size = size
        self.full_star = full_star
    def draw_star(self, canvas):
        if self.full_star:
            self.star = Image.open("user_data/star.png")
        else:
            self.star = Image.open("user_data/empty_star.png")
        self.star = self.star.resize((self.size, self.size), Image.ANTIALIAS)
        self.star = ImageTk.PhotoImage(self.star)
        canvas.create_image(self.x, self.y, anchor="c", image=self.star)
    def is_clicked(self, click_x, click_y):
        x0 = self.x - self.size//2
        y0 = self.y - self.size//2
        x1 = self.x + self.size//2
        y1 = self.y + self.size//2
        print(x0,y0,x1,y1, click_x, click_y)
        if x0 <= click_x <= x1 and y0 <= click_y <= y1:
            print("clicked")
            return True
        return False


        
## Variables

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
    data.mode = "login"
    data.cx, data.cy = data.width//2, data.height//2
    login_init(data)
    homescreen_init(data)
    camera_init(data)
    edit_init(data)
    movie_info_init(data)
    recommendations_init(data)
    
def login_init(data):
    login_button = Button("Login", size_to_canvas(data, (40,60,60,75)), 50, "Black")
    data.LOGIN_BUTTON = [login_button]
    data.username = ""
    data.app_name = Image.open("gui_files/name_small.png")
    data.app_name = data.app_name.resize((data.width//3, data.height//5))
    data.app_name = ImageTk.PhotoImage(data.app_name)
    data.current_user_id = None
    
def homescreen_init(data):
    photo_button = Button("Import Photo", size_to_canvas(data, (10,10,20,20)), 25)
    search_button = Button("Search", size_to_canvas(data, (80,10,90,20)), 25)
    camera_button = Button("Take Photo", size_to_canvas(data, (45,10,55,20)), 25)
    recommended_for_you_button = Button("Recommended For You", 
                                        size_to_canvas(data, (35,80,65,95)), 30)
    data.HOMESCREEN_BUTTON = [photo_button, search_button, 
                              recommended_for_you_button, camera_button]
    data.movie_title = ""

def camera_init(data):
    back_button = Button("Back", size_to_canvas(data, (5,5,15,15)), 25, "White")
    data.CAMERA_BUTTON = [back_button]
    
def edit_init(data):
    done_button = Button("Done", size_to_canvas(data, (10,10,20,20)), 25, "White")
    data.EDIT_BUTTON = [done_button]
    data.edit_poster = None
    data.bounding_boxes = None

def movie_info_init(data):
    home_button = Button("Home", size_to_canvas(data, (5,5,15,15)), 25)
    recommended_button = Button("Recommended Movies",
                                      size_to_canvas(data, (40,80,60,95)), 25)
    submit_button = Button("Submit", size_to_canvas(data, (90, 20, 95,25)), 15)
    data.MOVIE_INFO_BUTTON = [home_button, recommended_button, submit_button]
    def rating_system(data):
        data.display_star = Rating_Star((data.width-100,50), 60, True)
        data.rating_stars = []
        for i in range(5):
            sep = 30
            x = data.width - 150 + sep * i
            y = 100
            star = Rating_Star((x,y), 30, False)
            data.rating_stars.append(star)
        data.rating_open = False
    rating_system(data)

def recommendations_init(data):
    back_button = Button("Back", size_to_canvas(data, (5,5,15,15)), 20)
    data.RECOMMENDATIONS_BUTTON = [back_button]
    data.recommended_movies = []
    data.recommended_posters = []
    
## General

def mouse_pressed(event, data):
    if data.mode == "login": login_mouse_pressed(event, data)
    elif data.mode == "edit": edit_mouse_pressed(event, data)
    elif data.mode == "homescreen": homescreen_mouse_pressed(event, data)
    elif data.mode == "recommendations": recommendations_mouse_pressed(event, data)
    elif data.mode == "camera": camera_mouse_pressed(event, data)
    elif data.mode == "movie_info": movie_info_mouse_pressed(event, data)

def key_pressed(event, data):
    if data.mode == "login": login_key_pressed(event, data)
    elif data.mode == "edit": edit_key_pressed(event, data)
    elif data.mode == "homescreen": homescreen_key_pressed(event, data)
    elif data.mode == "recommendations": recommendations_key_pressed(event, data)
    elif data.mode == "camera": camera_key_pressed(event, data)
    elif data.mode == "movie_info": movie_info_key_pressed(event, data)

def redraw_all(canvas, data):
    if data.mode == "login": login_redraw_all(canvas, data)
    elif data.mode == "edit": edit_redraw_all(canvas, data)
    elif data.mode == "homescreen": homescreen_redraw_all(canvas, data)
    elif data.mode == "recommendations": recommendations_redraw_all(canvas, data)
    elif data.mode == "camera": camera_redraw_all(canvas, data)
    elif data.mode == "movie_info": movie_info_redraw_all(canvas, data)

def timer_fired(data):
    pass
    
## Login

def login_mouse_pressed(event, data):
    for button in data.LOGIN_BUTTON:
        if button.is_clicked(event.x, event.y):
            if button.name == "Login":
                if data.username != "":
                    initialize_login(data)
                    print("hello")
                    data.mode = "homescreen"

def initialize_login(data):
    users_data = pd.read_csv("user_data/user_data.csv")
    users = pd.read_csv("user_data/users.csv")
    if data.username not in users["username"].values:
        new_user_id = max(users_data["userId"]) + 1
        data.cur_user_id = new_user_id
        user = pd.DataFrame({"username": [data.username], "userId": [new_user_id]})
        users = users.append(user, ignore_index = True)
        user_data = dict()
        for col in users_data.columns:
            if col == "userId":
                user_data[col] = [new_user_id]
            else:
                user_data[col] = [0]
        user_data = pd.DataFrame(user_data)
        users_data = users_data.append(user_data, ignore_index = True)
        users_data.to_csv("user_data/user_data.csv", index = False)
        users.to_csv("user_data/users.csv", index = False)
    elif data.username in users["username"].values:
        data.cur_user_id = users[users["username"] == data.username].iloc[0, 1]
    data.users_data = users_data
    data.users = users
    
def login_redraw_all(canvas, data):
    #background
    canvas.create_rectangle(0, 0, data.width, data.height, fill="White")
    #logo
    canvas.create_image(data.width//2, data.height//5, anchor="c", image=data.app_name)
    #draws buttons
    for button in data.LOGIN_BUTTON:
        button.draw_button(canvas)
    #Input Line
    canvas.create_line(data.cx//2, data.cy, data.cx*3//2, data.cy, fill = "#E50914",
                       width=5)
    canvas.create_text(data.cx, data.cy-10, anchor = "s", text = data.username,
                       font="Arial 50 bold", fill="Black")
    
def login_key_pressed(event, data):
    print(event.keysym)
    if event.keysym == "BackSpace":
        if len(data.username) > 0:
            data.username = data.username[:-1]
    elif event.keysym in string.ascii_lowercase:
        data.username += event.keysym

## Homescreen

def homescreen_key_pressed(event, data):
    if event.keysym == "BackSpace":
        if len(data.movie_title) > 0:
            data.movie_title = data.movie_title[:-1]
    elif event.char in string.printable and event.keysym != "BackSpace":
        data.movie_title += event.char

def homescreen_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Black")
    for button in data.HOMESCREEN_BUTTON:
        button.draw_button(canvas)
    #draw input box
    canvas.create_line(data.cx//2, data.cy, data.cx*3//2, data.cy, fill = "#E50914",
                       width=5)
    canvas.create_text(data.cx, data.cy-10, anchor="s", text=data.movie_title,
                       font="Arial 50 bold", fill="White")

def homescreen_mouse_pressed(event, data):
    for button in data.HOMESCREEN_BUTTON:
        if button.is_clicked(event.x, event.y):
            if button.name == "Import Photo":
                data.edit_poster = Movie_Edit((data.cx, data.cy))
                data.edit_poster.resize(data.height*4//5)
                data.edit_poster.get_bounding_boxes()
                data.mode = "edit"
            elif button.name == "Search":
                if movie_info(data.movie_title) != False:
                    execute_search(data)
                    data.mode = "movie_info"
            elif button.name == "Recommended For You":
                user_liked_genres = get_liked_genres(data)
                rec_movies_id = movie_recommend(user_liked_genres)
                display_num = len(rec_movies_id) if len(rec_movies_id) < 4 else 4
                for i in range(display_num):
                    movie_id = rec_movies_id[i]
                    _, rec_movie = movie_info(movie_id)
                    data.recommended_movies.append(rec_movie)
                    rec_poster = Movie_Poster((data.width*(i+1)//5 + 15*i - 30, data.cy), rec_movie["poster_path"], rec_movie["original_title"])
                    rec_poster.resize(data.height//2)
                    data.recommended_posters.append(rec_poster)
                data.mode = "recommendations"
            elif button.name == "Take Photo":
                data.mode = "camera"
                
def get_liked_genres(data):
    binarize_thresh = 0.5
    id = data.cur_user_id 
    user_data_row = data.users_data[data.users_data["userId"] == id]
    thresh = binarize_thresh * max(user_data_row.iloc[0, 1:])
    genre_list = []
    for genre in user_data_row.columns[1:]:
        print(genre, user_data_row)
        if user_data_row.iloc[0][genre] >= thresh:
            genre_list.append(genre)
    print(genre_list)
    return genre_list
    
def execute_search(data):
    _, data.cur_movie_info = movie_info(data.movie_title)
    data.cur_poster = Movie_Poster((data.width//4, data.cy),
                                    data.cur_movie_info['poster_path'])
    data.cur_poster.resize(data.height*2//3)

## Camera

# Code from https://github.com/VasuAgrawal/112-opencv-tutorial/blob/master/opencvTkinterTemplate.py
def opencvToTk(frame):
    """Convert an opencv image to a tkinter image, to display in canvas."""
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    return tk_image

def camera_redraw_all(canvas, data):
    data.camera_frame = opencvToTk(data.frame)
    canvas.create_image(data.cx, data.cy, image=data.camera_frame)
    x, y, r = data.cx, data.height-100, 50
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="Grey")
    for button in data.CAMERA_BUTTON:
        button.draw_button(canvas)
    
def camera_mouse_pressed(event, data):
    x, y, r = data.cx, data.height-100, 50
    for button in data.CAMERA_BUTTON:
        if button.is_clicked(event.x, event.y):
            if button.name == "Back":
                data.mode = "homescreen"
    if in_distance(x, y, r, event.x, event.y):
        cv2.imwrite("frame.png", data.frame)
        data.mode = "homescreen"
    pass
    
def in_distance(x, y, r, click_x, click_y):
    if ((click_x - x)**2 + (click_y - y)**2)**(1/2) <= r:
        return True
    return False
    
def camera_key_pressed(event, data):
    pass

## Edit

def edit_key_pressed(event, data):
    for box in data.edit_poster.bounding_boxes:
        box.shift_side(event.keysym)

def edit_mouse_pressed(event, data):
    for button in data.EDIT_BUTTON:
        if button.is_clicked(event.x,event.y):
            if button.name == "Done":
                data.edit_poster.find_text()
                data.edit_poster.check_spelling()
                data.movie_title = data.edit_poster.movie_name
                data.mode = "homescreen"
    for box in data.edit_poster.bounding_boxes:
        box.is_clicked(event.x, event.y)
                
def edit_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Black")
    data.edit_poster.draw_poster(canvas)
    for button in data.EDIT_BUTTON:
        button.draw_button(canvas)

## Movie Info

def movie_info_mouse_pressed(event, data):
    if data.display_star.is_clicked(event.x, event.y):
        data.rating_open = not data.rating_open
    for button in data.MOVIE_INFO_BUTTON:
        if button.is_clicked(event.x, event.y):
            if button.name == "Home":
                data.mode = "homescreen"
            elif button.name == "Recommended Movies":
                rec_movies_id = movie_recommend(data.cur_movie_info["genre_ids"], 
                                                data.cur_movie_info["release_date"][:4])
                display_num = len(rec_movies_id) if len(rec_movies_id) < 4 else 4
                for i in range(display_num):
                    movie_id = rec_movies_id[i]
                    _, rec_movie = movie_info(movie_id)
                    data.recommended_movies.append(rec_movie)
                    rec_poster = Movie_Poster((data.width*(i+1)//5 + 15*i - 30, data.cy), rec_movie["poster_path"], rec_movie["original_title"])
                    rec_poster.resize(data.height//2)
                    data.recommended_posters.append(rec_poster)
                data.mode = "recommendations"
            elif data.rating_open:
                if button.name == "Submit":
                    score = 0
                    for star in data.rating_stars:
                        if star.full_star == True:
                            score += 1
                    if score >= 3:
                        for genre in data.genres:
                            data.users_data.loc[data.users_data["userId"] == data.cur_user_id,
                                                [genre]] += 1
                        data.users_data.to_csv("user_data/user_data.csv", index = False)
                            
                    data.rating_open = False
    for cur_star_index in range(len(data.rating_stars)):
        star = data.rating_stars[cur_star_index]
        if star.is_clicked(event.x, event.y):
            for i in range(len(data.rating_stars)):
                if i <= cur_star_index:
                    data.rating_stars[i].full_star = True
                else:
                    data.rating_stars[i].full_star = False
        
def movie_info_redraw_all(canvas, data):
    #background
    canvas.create_rectangle(0,0,data.width,data.height,fill="Black")
    #draw poster:
    data.cur_poster.draw_poster(canvas)
    #draw buttons
    for button in data.MOVIE_INFO_BUTTON:
        if button.name != "Submit":
            button.draw_button(canvas)
        elif data.rating_open:
            button.draw_button(canvas)
    #draw info
    score, data.genres, lines = draw_info(canvas, data)
    #draw rating system
    draw_rating_system(canvas, data, score)
    draw_genre_system(canvas, data, data.genres, lines)
    
def draw_genre_system(canvas, data, genres, lines):
    for i in range(len(genres)):
        genre = genres[i]
        x0 = data.width*3//7+90*i+10 
        y0 = data.height-330+lines*19
        x1 = data.width*3//7+90*(i+1)
        y1 = data.height-300+lines*19
        canvas.create_rectangle(x0,y0,x1,y1,outline="#E50914",width=3)
        canvas.create_text((x1-x0)//2+x0, (y1-y0)//2+y0, text=genre, fill="#E50914")
    
def draw_rating_system(canvas, data, score):
    data.display_star.draw_star(canvas)
    canvas.create_text(data.width-60, 50, anchor = "w", text=str(score), 
                       font="Arial 30 bold", fill="Grey")
    if data.rating_open:
        for star in data.rating_stars:
            star.draw_star(canvas)

def draw_info(canvas, data):
    score, title, lang, release_date, overview, genres = display_info(data.cur_movie_info)
    info_text = "Language: " + lang + "\nRelease Date: " + release_date + "\nOverview: \n" + overview
    canvas.create_text(data.width*3//7, data.height//4, anchor = "nw", text = info_text,
                       fill="White", font="Arial 20")
    canvas.create_text(data.width*3//7, data.height//6, anchor = "nw", text = title,
                       fill="White", font="Arial 40 bold")
    return score, genres, len(overview.split("\n"))

## Recommendation

def recommendations_mouse_pressed(event, data):
    for movie in data.recommended_posters:
        if movie.is_clicked(event.x, event.y):
            data.movie_title = movie.title
            if movie_info(data.movie_title) != False:
                execute_search(data)
                data.mode = "movie_info"
            
    for button in data.RECOMMENDATIONS_BUTTON:
        if button.is_clicked(event.x, event.y):
            if button.name == "Back":
                if data.movie_title != "":
                    data.mode = "movie_info"
                else:
                    data.mode = "homescreen"
def recommendations_redraw_all(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill="Black")
    for poster in data.recommended_posters:
        poster.draw_poster(canvas)
    for button in data.RECOMMENDATIONS_BUTTON:
        button.draw_button(canvas)
    
## Run Function
    
# Code from https://github.com/VasuAgrawal/112-opencv-tutorial/blob/master/opencvTkinterTemplate.py
#def camera_fired(data):
    data.frame = cv2.GaussianBlur(data.frame, (11, 11), 0)

# Taken from the 15-112 webpage
# Integrated with code from https://github.com/VasuAgrawal/112-opencv-tutorial/blob/master/opencvTkinterTemplate.py
def run(width=1000, height=600):
    # def redraw_all_wrapper(canvas, data):
    #     canvas.delete(ALL)
    #     canvas.create_rectangle(0, 0, data.width, data.height,
    #                             fill='white', width=0)
    #     redraw_all(canvas, data)
    #     canvas.update()    

    def mouse_pressed_wrapper(event, canvas, data):
        mouse_pressed(event, data)
        redraw_all_wrapper(canvas, data)

    def key_pressed_wrapper(event, canvas, data):
        key_pressed(event, data)
        redraw_all_wrapper(canvas, data)
    
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.camera_index = 0
    data.timer_delay = 100 # milliseconds
    data.redraw_delay = 50 # ms
    camera = cv2.VideoCapture(data.camera_index)
    #https://www.codingforentrepreneurs.com/blog/open-cv-python-change-video-resolution-or-scale/
    def make_1080p():
        camera.set(3, 1920)
        camera.set(4, 1080)
    make_1080p()
    data.camera = camera
    data.root = Tk()
    #creates Entry widget
    data.root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(data.root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    data.root.bind("<Button-1>", lambda event:
                            mouse_pressed_wrapper(event, canvas, data))
    data.root.bind("<Key>", lambda event:
                            key_pressed_wrapper(event, canvas, data))
                            
    def timer_fired_wrapper(canvas, data):
        start = time.time()
        timer_fired(data)
        redraw_all_wrapper(canvas, data)
        end = time.time()
        diff_ms = (end - start) * 1000
        delay = int(max(data.timer_delay - diff_ms, 0))
        # pause, then call timerFired again
        canvas.after(delay, timer_fired_wrapper(canvas, data))
    data.root.after(data.timer_delay, lambda: timer_fired_wrapper(canvas,data))
    def redraw_all_wrapper(canvas, data):
        start = time.time()
        if data.mode == "camera":
            _, data.frame = data.camera.read()
            #camera_fired(data)
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redraw_all(canvas, data)
        end = time.time()
        diff_ms = (end - start) * 1000
        delay = int(max(data.redraw_delay - diff_ms, 5))
        #data.root.after(delay, lambda: redraw_all_wrapper(canvas, data))
        canvas.update()    

    # and launch the app
    data.root.mainloop()  # blocks until window is closed
    data.camera.release()
    print("bye!")

run(1000, 600)