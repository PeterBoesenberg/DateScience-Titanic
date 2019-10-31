import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('train.csv')
df.columns
model = KNeighborsClassifier(n_neighbors = 10)
df = df.dropna()
x = df[['Age','Pclass']]

y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

model.fit(X_train, y_train)
model.score(X_test, y_test)
