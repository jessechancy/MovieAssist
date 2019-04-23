## Aim

"""
1. Grabs list from text_detect
2. Calls spell check on it if needed
3. Try to find it on tmdb
4. Puts into recommender
5. Tries to find recommendations in imdb
6. Sends to GUI
"""
import config # API key file
import requests
from text_detect_v2 import text_detect, spell_check
import json
from PIL import Image
from io import BytesIO
import urllib
import pandas as pd
from association_rule_mining import related_genres

#import movie_recommendation import recommend_movie

API_KEY = config.tmdb_api_key
#b8a019a6e0331c4c03fdc9202bc16777
#Action, Adventure, Animation, Comedy, Children, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
# GENRE_ID = {
#             28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy", 80:"Crime",
#             99:"Documentary", 18:"Drama", 10751:"Family", 14:"Fantasy", 
#             36:"History", 27:"Horror", 10402:"Music", 9648:"Mystery", 10749:"Romance",
#             878:"Science Fiction", 10770:"TV Movie", 53:"Thriller", 10752:"War",
#             37:"Western"
#             }
GENRES = {
            28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy", 80:"Crime",
            99:"Documentary", 18:"Drama", 10751:"Children", 14:"Fantasy", 
            27:"Horror", 10402:"Musical", 9648:"Mystery", 10749:"Romance",
            878:"Sci-Fi", 53:"Thriller", 10752:"War", 37:"Western"
            }
links_data = pd.read_csv("ml-latest-small/links.csv")
#Input Database movieId or movie name only
def movie_info(movie):
    if isinstance(movie, int):
        movie_id = str(int(links_data.loc[links_data["movieId"] == movie].iloc[0]["tmdbId"]))
        response = requests.get("https://api.themoviedb.org/3/movie/" + movie_id + "?api_key=" + API_KEY)
        response = response.json()
        movie = response["original_title"]
    movie_name = movie.replace(" ", "+")
    print(movie_name)
    response = requests.get('https://api.themoviedb.org/3/search/movie?api_key=' +  API_KEY + "&query=" + movie_name)
    try:
        response = response.json()
        total_results = response["total_results"]
        results = response["results"]
        top_result = results[0]
        return total_results, top_result # Change this later, now just return top result
    except:
        return False
        
    
#urllib taken from piazza
def movie_poster(poster_link):
    url = "http://image.tmdb.org/t/p/original" + poster_link
    response = urllib.request.urlretrieve(url, "tmp_search_poster.jpg")
    img = Image.open("tmp_search_poster.jpg")
    return img

#gets id list and gets recommended genres from rules
def movie_recommend(genre_id_list, movie_year):
    movies = pd.read_csv("ml-latest-small/movies.csv")
    genre_set = get_genres(genre_id_list)
    recommended_genres = related_genres(genre_set)
    possible_movie_list = []
    print(recommended_genres)
    movies["genres"] = movies["genres"].map(lambda s: set(s.split("|")))
    for related_genre in recommended_genres:
        complete_genre = related_genre.union(genre_set)
        movie_list = movies[movies["genres"] == (complete_genre)]
        print(movie_list)
        for index,row in movie_list.iterrows():
            print(row)
            movie_title = row["title"]
            movie_id = row["movieId"]
            try:
                date = int(movie_title[-5:-1])
                if abs(date - int(movie_year)) <= 10:
                    print("ehudshuheiuahiduehaiuheuhaiuehduiehaiuhdiuehauiede")
                    possible_movie_list.append(movie_id)
            except:
                continue
    return possible_movie_list
    
    #get recommended Movie
    #return recommended movie
    
def get_genres(genre_id_list):
    genre_set = set()
    for genre_id in genre_id_list:
        if genre_id in GENRES:
            genre_set.add(GENRES[genre_id])
    return genre_set
    

