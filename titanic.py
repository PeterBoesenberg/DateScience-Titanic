import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import operator

def handle_missing_and_categorial_values(df_missing):
    df_missing['Fare'].replace(np.NAN, df_missing['Fare'].mean(), inplace=True)
    df_missing['Embarked'] = df_missing['Embarked'].fillna(df_missing['Embarked'].mode())
    df_missing['Age'] = df_missing['Age'].fillna(df_missing['Age'].mean())
    df_missing['Sex'] = df_missing['Sex'].astype("category").cat.codes
    df_missing = pd.concat([df_missing, pd.get_dummies(df_missing['Embarked'], prefix='Embarked')], axis=1)
    return df_missing

def prepare_general_data():
    df = pd.read_csv('train.csv')
    df = handle_missing_and_categorial_values(df)
    df = df.drop(['PassengerId', 'Ticket', 'Cabin','Embarked'], axis=1)
    y = df['Survived']
    return df, y

def get_knn_score(X_train, X_test, y_train, y_test, neighbors=5):
    model = KNeighborsClassifier(n_neighbors=neighbors)
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

knn_scores = {}

def explore_knn():
    get_all_knn_scores()
    print(knn_scores)
    maximum = max(knn_scores.items(), key=operator.itemgetter(1))
    print(maximum[0], maximum[1])

def read_test_data():
    df_test = pd.read_csv('test.csv')
    df_test = handle_missing_and_categorial_values(df_test)
    return df_test


def create_prediction_csv(df_test, model):
    df_test = df_test[['PassengerId', 'Survived']]
    df_test.to_csv(path_or_buf='result.csv', index=False, float_format='%.f')


def create_knn_prediction_csv():
    X_train, X_test, y_train, y_test = set_features(['Fare'])
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)
    df_test = read_test_data()
    df_test['Survived'] = model.predict(df_test[['Fare']])
    create_prediction_csv(df_test, model)


def create_random_forest_csv():
    features = ["Age", "Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]
    X_train, X_test, y_train, y_test = set_features(features)

    model = RandomForestClassifier(
        n_estimators=32, max_depth=6, min_samples_split=0.2, min_samples_leaf=0.2, random_state=1)
    model.fit(X_train, y_train)
    df_test = read_test_data()
    df_test['Survived'] = model.predict(df_test[features])
    create_prediction_csv(df_test, model)

def create_logistic_regression_csv():
    features = [ "Fare", "Sex","Age" ,"Embarked_C", "Embarked_Q", "Embarked_S"]
    X_train, X_test, y_train, y_test = set_features(features)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    df_test = read_test_data()
    df_test['Survived'] = model.predict(df_test[features])
    create_prediction_csv(df_test, model)

def create_xgboost_csv():
    features = ["Age", "Sex"]
    X_train, X_test, y_train, y_test = set_features(features)
   

    model = xgb.XGBRegressor(
        objective = 'binary:hinge'
    ) 

    model.fit(X_train, y_train)
    
    df_test = read_test_data()
    df_test['Survived'] = model.predict(df_test[features])
    create_prediction_csv(df_test, model)

df.info()
print(df.head())
create_logistic_regression_csv()