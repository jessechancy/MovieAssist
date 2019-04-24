## Aim
"""
Mines Association Rules within user ratings data
    transaction: set of genres a user likes
    items: genres

1. Reformats the provided dataset into a dataframe of users with their 
   rated genre scores (Use Once)
2. Binarizes this rated genre score to True or False
3. Apropri Algorithm to get frequent subsets. Means there is enough data to 
   support an association
4. Generates Rules based on frequent subsets
5. Find related genres based on rules

"""

## Imports

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules

## Set Thresholds

support_thresh = 0.3
confidence_thresh = 0.5
lift_thresh = 1
rating_thresh = 3 #out of 5
#percentage of maximum where count >= binarize_thresh is consider liking a genre
binarize_thresh = 0.5 

## Paths

user_data_path = "user_data/user_data.csv"
user_data_binarized_path = "user_data/user_data_binarized.csv"

## Read Data

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

## Reformat Data

#binarizes genre data to each movie
def concat_data(movies, ratings):
    mlb = MultiLabelBinarizer()
    #breaks up string genres into lists of genres
    movies["genres"] = movies["genres"].map(lambda s: set(s.split("|")))
    #generate dataframe with binarized values
    binarized = mlb.fit_transform(movies["genres"])
    movie_genres = pd.DataFrame(binarized, columns = mlb.classes_)
    movie_data = pd.concat([movies["movieId"], movie_genres], axis=1)
    #get highly rated movies by users and add them to the movies
    ratings_data = get_ratings_data()
    movie_ratings_data = pd.merge(movie_data, ratings_data, on="movieId")
    return movie_ratings_data

#get highly rated movies by users
def get_ratings_data():
    highly_rated = ratings[ratings["rating"] >= rating_thresh]
    ratings_data = highly_rated.filter(["movieId", "userId"], axis=1)
    return ratings_data

def reformat_data(movie_ratings_data):
    genre_list = get_genre_list(movie_ratings_data)
    #gets a dataframe of distinct userId
    unique_users_list = movie_ratings_data["userId"].unique()
    unique_users_df = pd.DataFrame(unique_users_list, columns = ["userId"])
    #generates an empty genre base for each user
    user_num = len(unique_users_list)
    user_genre_base = pd.DataFrame([{i : 0 for i in genre_list}] * user_num)
    user_genre_data = pd.concat((unique_users_df, user_genre_base), axis=1)
    count = 0
    for user in unique_users_list:
        count += 1
        print("loading " + str(count) + " of " + str(user_num))
        #gets a dataframe of movies user rated highly
        user_data = movie_ratings_data[movie_ratings_data["userId"] == user]
        for _, row in user_data.iterrows():
            for genre in genre_list:
                if row[genre] == 1: #rated highly
                    user_genre_data.loc[user_genre_data["userId"] == user, [genre]] += 1
    return user_genre_data
    
def get_genre_list(movie_ratings_data):
    #creates a list of genres
    genre_list = list(movie_ratings_data)
    genre_list = remove_unwanted_columns(genre_list)
    return genre_list

def remove_unwanted_columns(headings):
    unwanted_cols = {'movieId', '(no genres listed)', 'IMAX', 'userId'}
    new_headings = []
    for col in headings:
        if col not in unwanted_cols:
            new_headings.append(col)
    return new_headings

# Does not return anything (USE ONCE)
def reformat_data_main(path):
    movie_ratings_data = concat_data(movies, ratings)
    user_genre_data = reformat_data(movie_ratings_data)
    user_genre_data.to_csv(path, index = False)
    
## Binarize User Data

def binarize_user_data(path):
    user_genre_data = pd.read_csv(path)
    genre_list = get_genre_list(user_genre_data)
    user_genre_data = pd.read_csv(path)
    rows, _ = user_genre_data.shape
    for row in range(rows):
        user_genre_count = user_genre_data.iloc[row, 1:] #ignore movieId column
        threshold_max = binarize_thresh * max(user_genre_count)
        for genre in genre_list:
            if user_genre_data.iloc[row][genre] >= threshold_max:
                user_genre_data.loc[[row], [genre]] = True
            else:
                user_genre_data.loc[[row], [genre]] = False
    return user_genre_data

def binarize_user_data_main(path):
    user_genre_data = binarize_user_data(user_data_path)
    user_genre_data.to_csv(path, index = False)
    
## Frequent Itemset Generation

def generate_frequent_itemset(path):
    movie_data = pd.read_csv(path)
    movie_data = movie_data.iloc[:, 1:]
    frequent_item_set = apriori(movie_data, min_support=support_thresh, use_colnames=True)
    return frequent_item_set

## Rule Generation

def rule_generation(frequent_itemsets):
    rules = association_rules(frequent_itemsets, metric = "confidence",
                              min_threshold=confidence_thresh)
    return rules

## Main Run Function

def generate_association_rules():
    try:
        frequent_itemsets = generate_frequent_itemset(user_data_binarized_path)
        association_rules = rule_generation(frequent_itemsets)
    except:
        reformat_data_main(user_data_path)
        binarize_user_data_main(user_data_binarized_path)
        frequent_itemsets = generate_frequent_itemset(user_data_binarized_path)
        association_rules = rule_generation(frequent_itemsets)
    return association_rules
