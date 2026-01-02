import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(accuracy_score(y_test, predict) * 100)

model1 = LogisticRegression(max_iter=200)
model1.fit(X_train, y_train)
predict1 = model1.predict(X_test)
print(accuracy_score(y_test, predict1) * 100)

model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
predict3 = model2.predict(X_test)
print(accuracy_score(y_test, predict3) * 100)
