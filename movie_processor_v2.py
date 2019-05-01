## Aim

"""
Functions:
1. Get Movie Info
2. Get Movie Poster
3. Recommend Movie Based on Rules
"""

## Imports

import config #API_KEY file
import requests
from text_detect_v2 import text_detect, spell_check
import json
from PIL import Image
import urllib
import pandas as pd
from association_rule_mining_v2 import generate_association_rules

## API_KEY

API_KEY = config.tmdb_api_key

## Genres

GENRES = {
            28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy", 80:"Crime",
            99:"Documentary", 18:"Drama", 10751:"Children", 14:"Fantasy", 
            27:"Horror", 10402:"Musical", 9648:"Mystery", 10749:"Romance",
            878:"Sci-Fi", 53:"Thriller", 10752:"War", 37:"Western"
            }

## Read Data

links = pd.read_csv("ml-latest-small/links.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

## 1. Movie Info

#input movieId or movie name
def movie_info(movie):
    if isinstance(movie, int):
        movie = find_movie(movie)
        if movie == False:
            return False
    movie_name = movie.replace(" ", "+")
    response = requests.get('https://api.themoviedb.org/3/search/movie?api_key=' +  API_KEY + "&query=" + movie_name)
    try:
        response = response.json()
        result_count = response["total_results"]
        results_list = response["results"]
        #get result
        top_result = most_popular(results_list)
        return result_count, top_result
    except:
        return False
        
def most_popular(results_list):
    threshold = 1/5
    high_vote_count = 0
    high_result = None
    for result in results_list:
        vote_count = result["vote_count"]
        if vote_count > high_vote_count:
            high_vote_count = vote_count
            high_result = result
    return high_result
        
#takes in int movieId and finds movie name
def find_movie(movie):
    tmdb_movie_id = str(int(links.loc[links["movieId"] == movie].iloc[0]["tmdbId"]))
    response = requests.get("https://api.themoviedb.org/3/movie/" + tmdb_movie_id + "?api_key=" + API_KEY)
    try:
        response = response.json()
        movie = response["original_title"]
        return movie
    except:
        return False

## 2. Movie Poster

def movie_poster(poster_link):
    url = "http://image.tmdb.org/t/p/original" + poster_link
    response = urllib.request.urlretrieve(url, "tmp_search_poster.jpg")
    img = Image.open("tmp_search_poster.jpg")
    return img

## 3. Recommend Movie

# Set Recommendation Parameters
lift_thresh = 1
year_thresh = 5

def movie_recommend(movie_genres, movie_year="2019"):
    if isinstance(movie_genres[0], int):
        movie_genres = get_genres(movie_genres)
    else:
        movie_genres = set(movie_genres)
    rules = generate_association_rules()
    recommended_genres = related_genres(movie_genres, rules)
    if not isinstance(movies["genres"][0], set):
        movies["genres"] = movies["genres"].map(lambda s: set(s.split("|")))
    movie_list = get_movie_list(movie_year, recommended_genres, movie_genres)
    return movie_list

def get_movie_list(movie_year, recommended_genres, movie_genres):
    movie_list = []
    print(recommended_genres)
    for genre_rec in recommended_genres:
        genre_rec = genre_rec.union(movie_genres)
        recommended_movies = movies[movies["genres"] == (genre_rec)]
        if len(recommended_movies) == 0:
            empty = True
            subsets_of_genre_rec = [genre_rec]
            while empty:
                genre_subset = []
                
                for genre_set in subsets_of_genre_rec:
                    genre_subset.extend(subset(genre_set))
                for sub in genre_subset[:2]:
                    recommended_movies = movies[movies["genres"] == sub]
                    if len(recommended_movies) != 0:
                        print("hello")
                        empty = False
                        break
                subsets_of_genre_rec = genre_subset
                print(empty)
        print("passed!", recommended_movies)
        for _, row in recommended_movies.iterrows():
            movie_title = row["title"]
            movie_id = row["movieId"]
            try:
                date = int(movie_title[-5:-1])
                if abs(date - int(movie_year)) <= year_thresh:
                    movie_list.append(movie_id)
            except:
                continue
    return movie_list
    
#returns dataframe of consequents of rules
def related_genres(genre_set, rules):
    if len(genre_set) == 0:
        return None
    #filter out antecedent rules that are equal to genre set
    ar_genre = rules[rules["antecedents"] == genre_set]
    #lift filter
    ar_genre_lift = ar_genre[ar_genre["lift"] >= lift_thresh]
    #return the related sets
    result = ar_genre_lift["consequents"].tolist()
    if len(result) == 0: 
        for genres in subset(genre_set):
            tmp_result = related_genres(genres, rules)
            if tmp_result != None:
                result += tmp_result
    if len(result) != 0:
        return result

def subset(genre_set):
    set_length = len(genre_set) - 1
    #possible n-1 subsets of a set is set length
    combinations = len(genre_set)
    genre_list = list(genre_set)
    subset_list = []
    for i in range(combinations):
        if i+set_length <= combinations:
            subset_list.append(set(genre_list[i:i+set_length]))
        else:
            subset_list.append(set(genre_list[i:] + genre_list[:(i+set_length)%combinations]))
    return subset_list
    
#returns genre name from genre id if genre recognized
def get_genres(genre_id_list):
    genre_set = set()
    for genre_id in genre_id_list:
        if genre_id in GENRES:
            genre_set.add(GENRES[genre_id])
    return genre_set

## 4. Display Info

def display_info(input_movie):
    score = input_movie["vote_average"]
    title = input_movie["original_title"]
    lang = input_movie["original_language"]
    release_date = input_movie["release_date"]
    overview = reformat_text(input_movie["overview"], 50)
    reformat_overview = overview.split("\n")
    overview_lines = len(reformat_overview)
    if overview_lines > 10:
        reformat_overview[9] = reformat_overview[9] + " ... "
        overview = "\n".join(reformat_overview[:10])
    genre_ids = input_movie["genre_ids"]
    genres = []
    for id in genre_ids:
        if id in GENRES:
            genres.append(GENRES[id])
    return score, title, lang, release_date, overview, genres
    
#cite: myself in hw3
def reformat_text(text, text_width):
    white_space_index = 0
    index = 0
    while index < len(text):
        index = white_space_index
        width_limit = text_width + white_space_index + 1
        while index <= width_limit and index < len(text):
            if text[index] == " ":
                white_space_index = index
            index += 1
        if index < len(text):
            text = text[:white_space_index] + "\n" + text[white_space_index + 1:]
    return text
    