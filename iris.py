import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
%matplotlib inline
import seaborn as sns
iris=load_iris()
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
y_test
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
predict=model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predict)*100)
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression(max_iter=100)
model1.fit(X_train,y_train)
predict1=model1.predict(X_test)
predict1
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predict)*100)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predict3)*100)
print(accuracy_score(y_test,predict)*100)
from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()
model2.fit(X_train,y_train)
predict3=model1.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predict3)*100)
