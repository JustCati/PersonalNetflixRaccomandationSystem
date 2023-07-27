import os
import numpy as np
import pandas as pd



def buildVector(user, ratings, numMovies):
    toRet = np.zeros(numMovies, dtype=np.int32)
    userRatings = ratings[ratings.User_ID == user]
    for _, row in userRatings.iterrows():
        toRet[row.Movie_ID] = row.Rating
    return toRet


def createVectorizedDataset(currentPath=os.path.dirname(__file__)):
    ratingsPath = os.path.join(currentPath, ".datasetCache", "ratings.parquet")

    dfRatings = pd.read_parquet(ratingsPath)
    users = list(set(dfRatings.User_ID.values))
    numMovies = len(set(dfRatings.Movie_ID.values))

    df = pd.DataFrame(users, columns=["User_ID"])
    df["Ratings"] = [np.zeros(numMovies, dtype=np.int32) for _ in range(len(users))]
    df.Ratings = df.apply(lambda x: buildVector(x.User_ID, dfRatings, numMovies), axis=1)

    return df


def getUtilityMatrix(currentPath=os.path.dirname(__file__)):
    df = createVectorizedDataset()
    
    matrix = pd.DataFrame(index=df.User_ID, columns=range(len(df.Ratings[0])))
    for i in range(len(df.Ratings)):
        matrix.iloc[i] = df.Ratings[i]

    moviesPath = os.path.join(currentPath, ".datasetCache", "moviesIDS.parquet")
    movieIDS = pd.read_parquet(moviesPath)
    matrix = matrix.rename(columns=movieIDS.set_index("Movie_ID").Name.to_dict())
    return matrix

