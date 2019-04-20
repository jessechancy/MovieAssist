## Aim for this Script
# Take in a movie title
# Recommend base on movie title, which you search for genre in database

import config
import requests
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split

# api_key = config.tmdb_api_key
# movie_name = "Jack+Reacher"
# response = requests.get('https://api.themoviedb.org/3/search/movie?api_key=' +  api_key + "&query=" + movie_name)
# print(response.text)

## Import CSV Files
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
links = pd.read_csv("ml-latest-small/links.csv")

## CSV File Pre-Processing

def get_ending_date(s):
    flagged = False
    index = -1
    while not flagged:
        if index == -3:
            flagged == True
        try:
            int(s[index])
            flagged = True
        except:
            index -= 1
    return s[index-4:index+1]

#splits genres into a list
movies["genres"] = movies["genres"].map(lambda s: s.split("|"))
#movie_dates = movies["title"].map(lambda s: get_ending_date(s))

mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform(movies["genres"])
movie_genres = pd.DataFrame(binarized, columns = mlb.classes_)
#movie_data = pd.concat([movies["movieId"], movie_dates, movie_genres], axis=1)
movie_data = pd.concat([movies["movieId"], movie_genres], axis=1)
movie_data.to_csv("final2.csv", sep='\t', encoding='utf-8')

## Feature Selection

#features = ["title"] + list(mlb.classes_)
features = list(mlb.classes_)
f_len = len(features)
X = movie_data[features]
y = movie_data["movieId"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# scale_data = MinMaxScaler()
# X_train = scale_data.fit_transform(X_train)
# X_test = scale_data.transform(X_test)

## Machine Learning Recommender

mlp_classifier = MLPClassifier(hidden_layer_sizes = (f_len,f_len,f_len))
trained = mlp_classifier.fit(X_train, y_train)