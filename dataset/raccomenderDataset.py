import os
import numpy as np
import pandas as pd



def createRatingDataset(df, currentPath=os.path.dirname(__file__)):
    datasetPath = os.path.join(currentPath, ".cache")

    movies = pd.read_csv(os.path.join(datasetPath, "Netflix_Dataset_Movie.csv"))
    movies = movies[movies.Name.isin(df.Titolo.values)]

    ratings = pd.read_csv(os.path.join(datasetPath, "Netflix_Dataset_Rating.csv"))
    ratings = ratings[ratings.Movie_ID.isin(movies.Movie_ID.values)]

    res = ratings.merge(movies, on="Movie_ID")
    res = res.drop(columns=["Movie_ID", "Year"])

    ids = pd.DataFrame({"Name": sorted(res.Name.unique()), "Movie_ID": range(len(res.Name.unique()))})
    ids.to_parquet(os.path.join(currentPath, ".datasetCache", "moviesIDS.parquet"))
    
    res.Name = res.apply(lambda x: ids[ids.Name == x.Name].Movie_ID.values[0], axis=1)
    res = res.rename(columns={"Name": "Movie_ID"})

    res.to_parquet(os.path.join(currentPath, ".datasetCache", "ratings.parquet"))
    return res


def buildVector(user, ratings, numMovies):
    toRet = np.zeros(numMovies, dtype=np.int32)
    userRatings = ratings[ratings.User_ID == user]
    for _, row in userRatings.iterrows():
        toRet[row.Movie_ID] = row.Rating
    return toRet


def createVectorizedDataset(movies, currentPath=os.path.dirname(__file__)):
    ratingsPath = os.path.join(currentPath, ".datasetCache", "ratings.parquet")

    if not os.path.exists(ratingsPath):
        dfRatings = createRatingDataset(movies)
        dfRatings.to_parquet(ratingsPath)
    else:
        dfRatings = pd.read_parquet(ratingsPath)

    users = list(set(dfRatings.User_ID.values))
    numMovies = len(set(dfRatings.Movie_ID.values))

    df = pd.DataFrame(users, columns=["User_ID"])
    df["Ratings"] = [np.zeros(numMovies, dtype=np.int32) for _ in range(len(users))]
    df.Ratings = df.apply(lambda x: buildVector(x.User_ID, dfRatings, numMovies), axis=1)

    return df


def getUtilityMatrix(movies, currentPath=os.path.dirname(__file__)):
    df = createVectorizedDataset(movies)
    
    matrix = pd.DataFrame(index=df.User_ID, columns=range(len(df.Ratings[0])))
    for i in range(len(df.Ratings)):
        matrix.iloc[i] = df.Ratings[i]

    moviesPath = os.path.join(currentPath, ".datasetCache", "moviesIDS.parquet")
    movieIDS = pd.read_parquet(moviesPath)

    matrix = matrix.rename(columns=movieIDS.set_index("Movie_ID").Name.to_dict())
    return matrix
