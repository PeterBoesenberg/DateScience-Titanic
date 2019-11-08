import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



def prepare_general_data():
    df = pd.read_csv('train.csv')
    df = df.dropna()
    df['Sex'] = df['Sex'].astype("category").cat.codes
    y = df['Survived']
    return df, y

def get_knn_score(X_train, X_test, y_train, y_test,neighbors = 10):
    model = KNeighborsClassifier(n_neighbors = neighbors)
    model.fit(X_train, y_train)
    knn_score = model.score(X_test, y_test)
    return knn_score
    
def set_features(features):
    x = df[features]
    return train_test_split(x, y, random_state=0)

def get_single_column_knn_score(column):
    X_train, X_test, y_train, y_test = set_features(column)
    score = get_knn_score(X_train, X_test, y_train, y_test)
    knn_scores.update({', '.join(column): score})


def get_single_column_knn_scores():
    get_single_column_knn_score(['Age'])
    get_single_column_knn_score(['Pclass'])
    get_single_column_knn_score(['SibSp'])
    get_single_column_knn_score(['Parch'])
    get_single_column_knn_score(['Fare'])
    get_single_column_knn_score(['Sex'])
    get_single_column_knn_score(['Age', 'Fare'])
    get_single_column_knn_score(['SibSp', 'Parch'])
    get_single_column_knn_score(['Age', 'Parch'])
    get_single_column_knn_score(['Fare', 'Parch'])
    get_single_column_knn_score(['Sex', 'Fare'])
    get_single_column_knn_score(['Sex','Pclass'])


df, y = prepare_general_data()
# X_train, X_test, y_train, y_test = set_features()

knn_scores = {}
get_single_column_knn_scores()
print(knn_scores)

