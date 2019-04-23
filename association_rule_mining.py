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
"""
1. Convert User data aggregate genre points
2. Redistribute points
3. Pass into apropri function
4. Generate output
"""

## Imports

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

## Set Thresholds

MIN_SUPPORT = 10
MIN_CONFIDENCE = 0.7

## Get Data

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies["genres"] = movies["genres"].map(lambda s: set(s.split("|")))
movie_dates = movies["title"].map(lambda s: s[-5:-1])
mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform(movies["genres"])
movie_genres = pd.DataFrame(binarized, columns = mlb.classes_)
#movie_data = pd.concat([movies["movieId"], movie_dates, movie_genres], axis=1)
movie_data = pd.concat([movies["movieId"], movie_genres], axis=1)
highly_rated = ratings[ratings["rating"] >= 3]
ratings_data = highly_rated.filter(["movieId", "userId"], axis=1)
movie_ratings_data = pd.merge(movie_data, ratings_data, on="movieId")
genre_list = list(movie_ratings_data)
genre_list.remove("movieId")
genre_list.remove("userId")
genre_list.remove("(no genres listed)")

###################
# Format Data
###################

## Translate to User_data

def get_user_genre_data(movies, ratings):
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
    genre_list = list(movie_ratings_data)
    genre_list.remove("movieId")
    genre_list.remove("userId")
    genre_list.remove("(no genres listed)")
    #all the different users
    distinct_users = movie_ratings_data["userId"].unique()
    distinct_user_df = pd.DataFrame(distinct_users, columns = ["userId"])
    #assign each user a genre score
    user_num = len(distinct_users)
    one_user_genre_base = {i : 0 for i in genre_list}
    all_user_genre_base = [one_user_genre_base] * user_num
    user_genre_data = pd.DataFrame(all_user_genre_base)
    user_genre_data = pd.concat((distinct_user_df, user_genre_data), axis=1)
    count = 0
    for user in movie_ratings_data["userId"].unique():
        count += 1
        print(count)
        user_data = movie_ratings_data[movie_ratings_data["userId"] == user]
        for index, row in user_data.iterrows():
            for genre in genre_list:
                if row[genre] == 1:
                    user_genre_data.loc[user_genre_data["userId"] == user, [genre]] += 1
                    print(user_genre_data.loc[user_genre_data["userId"] == user, [genre]])
    user_genre_data.to_csv("user_data.csv")

## Open User data 
path = "user_data/user_data.csv"
def binarize_user_data(path):
    threshold = 0.5
    user_genre_data = pd.read_csv(path)
    rows, _ = user_genre_data.shape
    for row in range(rows):
        user_genres = user_genre_data.iloc[row, 2:]
        print(user_genres)
        user_max = max(user_genres)
        print(user_max)
        threshold_max = 0.3 * user_max
        for genre in genre_list:
            if user_genre_data.iloc[row][genre] >= threshold_max:
                user_genre_data.loc[[row], [genre]] = True
            else:
                user_genre_data.loc[[row], [genre]] = False
    user_genre_data.to_csv("user_data/user_data_changed.csv")
#binarize_user_data(path)

## Frequent Item Set Generation (Using Apropri)

def apropri_use():
    movie_data = pd.read_csv("user_data/user_data_changed.csv")
    movie_data = movie_data.iloc[:, 3:]
    frequent_item_set = apriori(movie_data, min_support=0.5, use_colnames=True)
    return frequent_item_set
    
## Rule Generation

def rule_generation(frequent_itemsets):
    rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold=0.8)
    rules.to_csv("tmp.csv")
    return rules
    
## 

def related_genres(genre_set):
    frequent_item_set = apropri_use()
    rules = rule_generation(frequent_item_set)
    ar_genre = rules[rules["antecedents"] == genre_set]
    ar_genre_lift = ar_genre[ar_genre["lift"] >= 1]
    return ar_genre_lift["consequents"]

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
#https://hub.packtpub.com/article-movie-recommendation/
# def rule_generation(frequent_itemsets):
#     candidate_rules = []
#     for itemset_length, itemset_counts in frequent_itemsets.items():
#         for itemset in itemset_counts.keys():
#             for conclusion in itemset:
#                 premise = itemset - set((conclusion,))
#                 candidate_rules.append((premise, conclusion))
#     
