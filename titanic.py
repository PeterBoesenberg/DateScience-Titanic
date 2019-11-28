import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import operator


def prepare_general_data():
    df = pd.read_csv('train.csv')
    df = df.dropna()
    df['Sex'] = df['Sex'].astype("category").cat.codes
    df['Embarked'] = df['Embarked'].astype("category").cat.codes
    y = df['Survived']
    return df, y

def get_knn_score(X_train, X_test, y_train, y_test,neighbors = 5):
    model = KNeighborsClassifier(n_neighbors = neighbors)
    model.fit(X_train, y_train)
    knn_score = model.score(X_test, y_test)
    return knn_score
    
def set_features(features):
    x = df[features]
    return train_test_split(x, y, random_state=0)

def get_one_knn_scores(column):
    X_train, X_test, y_train, y_test = set_features(column)
    score = get_knn_score(X_train, X_test, y_train, y_test)
    knn_scores.update({', '.join(column): score})


def get_all_knn_scores():
    get_one_knn_scores(['Age'])
    get_one_knn_scores(['Pclass'])
    get_one_knn_scores(['SibSp'])
    get_one_knn_scores(['Parch'])
    get_one_knn_scores(['Fare'])
    get_one_knn_scores(['Sex'])
    get_one_knn_scores(['Embarked'])
    get_one_knn_scores(['Age', 'Pclass'])
    get_one_knn_scores(['Age', 'SibSp'])
    get_one_knn_scores(['Age', 'Parch'])
    get_one_knn_scores(['Age', 'Fare'])
    get_one_knn_scores(['Age', 'Sex'])
    get_one_knn_scores(['Age', 'Embarked'])
    get_one_knn_scores(['Pclass', 'SibSp'])
    get_one_knn_scores(['Pclass', 'Parch'])
    get_one_knn_scores(['Pclass', 'Fare'])
    get_one_knn_scores(['Pclass', 'Sex'])
    get_one_knn_scores(['Pclass', 'Embarked'])
    get_one_knn_scores(['Parch', 'SibSp'])
    get_one_knn_scores(['Parch', 'Fare'])
    get_one_knn_scores(['Parch', 'Sex'])
    get_one_knn_scores(['Parch', 'Embarked'])
    get_one_knn_scores(['SibSp', 'Fare'])
    get_one_knn_scores(['SibSp', 'Sex'])
    get_one_knn_scores(['SibSp', 'Embarked'])
    get_one_knn_scores(['Fare', 'Sex'])
    get_one_knn_scores(['Fare', 'Embarked'])
    get_one_knn_scores(['Sex', 'Embarked'])


df, y = prepare_general_data()

def explore_knn():
    knn_scores = {}
    get_all_knn_scores()
    print(knn_scores)
    maximum = max(knn_scores.items(), key=operator.itemgetter(1))
    print(maximum[0],maximum[1])


def create_knn_prediction_csv():
    X_train, X_test, y_train, y_test = set_features(['Fare'])
    model = KNeighborsClassifier(n_neighbors = 9)
    model.fit(X_train, y_train)

    df_test = pd.read_csv('test.csv')
    df_test['Fare'].replace(np.NAN,0, inplace=True)
    df_test['Sex'] = df_test['Sex'].astype("category").cat.codes
    df_test['Survived'] = model.predict(df_test[['Fare']])
    df_test = df_test[['PassengerId','Survived']]
    df_test.to_csv(path_or_buf='result.csv',index=False)

create_knn_prediction_csv()
