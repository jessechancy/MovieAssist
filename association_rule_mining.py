##########################
# Basic Recommender (Placeholder)
##########################
"""
1. Takes in a movie with differnet genres
"""

def recommend_movie(movie):
    return movie

## Aim

"""
1. Mine the association rules given a set of user data
1. items = articles
2. Transactions = users

In regular case, transaction is bread, butter, items are bread and butter

1. User = user recommendations (640)
    User like movie with Romance, comedy ==> they also like movies with ...
2. Item = a sum of genres 101110101 ==> you also like 10110101
"""

## Imports

from mlxtend.frequent_patterns import apriori
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

## Set Thresholds

MIN_SUPPORT = 5
MIN_CONFIDENCE = 0.7

## Get Data

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

## Format Data

movies["genres"] = movies["genres"].map(lambda s: s.split("|"))
movie_dates = movies["title"].map(lambda s: s[-5:-1])
mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform(movies["genres"])
movie_genres = pd.DataFrame(binarized, columns = mlb.classes_)
#movie_data = pd.concat([movies["movieId"], movie_dates, movie_genres], axis=1)
movie_data = pd.concat([movies["movieId"], movie_genres], axis=1)
highly_rated = ratings[ratings["rating"] >= 3]
ratings_data = highly_rated.filter(["movieId", "userId"], axis=1)
movie_ratings_data = pd.merge(movie_data, ratings_data, on="movieId")
movie_ratings_data = movie_ratings_data.groupby("userId").agg("|".join)


## Frequent Item Set Generation (Using Apropri)

#frequent_item_set = apriori(movie_data, min_support=0.6, use_colnames=True)

##########################
# Frequent Item Set Generation
##########################

## Reduce Number of Candidate Itemsets

# def apriopri_algorithm():
#     item = 1
#     frequent_itemset = get_frequent_itemset(item, data)
#     while len(frequent_itemset) != 0:
#         item += 1
#         candidate_itemset = get_candidate_itemset(frequent_itemset)
#         for t in transactions:
#             candidate_itemset_transaction = get_subsets(candidate_itemset, t)
#             for c in candidate_itemset_transaction:
#                 support_count_c += 1
#         frequent_itemset = get_frequent_itemset()
#     return all_frequent_itemset
# 
# def get_frequent_itemset(item):
#     frequent_itemset = set()
#     for i in genres:
#         if i//len(data) >= min_support:
#             frequent_itemset.add(i)
#     return frequent_itemset

## Reduce the Number of Comparisons



##########################
# Rule Generation
##########################

