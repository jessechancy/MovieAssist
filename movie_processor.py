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
#import movie_recommendation import recommend_movie

API_KEY = config.tmdb_api_key
#b8a019a6e0331c4c03fdc9202bc16777

def movie_info(movie):
    movie_name = movie.replace(" ", "+")
    response = requests.get('https://api.themoviedb.org/3/search/movie?api_key=' +  API_KEY + "&query=" + movie_name)
    try:
        response = response.json()
        total_results = response["total_results"]
        results = response["results"]
        top_result = results[0]
        return total_results, top_result # Change this later, now just return top result
    except:
        return False
