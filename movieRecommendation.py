## Aim for this
# Take in a movie title
# Recommend base on movie title, which you search for genre in database

import config
import requests
import pandas as pd
from sklearn.neural_network import MLPClassifier


api_key = config.tmdb_api_key
movie_name = "Jack+Reacher"
response = requests.get('https://api.themoviedb.org/3/search/movie?api_key=' +  api_key + "&query=" + movie_name)
print(response.text)
# 
# movies = pd.read_csv("ml-latest-small/")
