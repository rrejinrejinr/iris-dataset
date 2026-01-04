import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.title("Iris Flower Classification")

model_name = st.selectbox(
    "Select Model",
    ("SVM", "Logistic Regression", "Decision Tree")
)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if model_name == "SVM":
    model = SVC()
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
else:
    model = DecisionTreeClassifier()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.write("Accuracy:", round(accuracy * 100, 2), "%")

st.subheader("Predict a Flower")

sepal_length = st.number_input(
    "Sepal Length", min_value=4.0, max_value=8.0, value=5.1, step=0.1
)

sepal_width = st.number_input(
    "Sepal Width", min_value=2.0, max_value=4.5, value=3.5, step=0.1
)

petal_length = st.number_input(
    "Petal Length", min_value=1.0, max_value=7.0, value=1.4, step=0.1
)

petal_width = st.number_input(
    "Petal Width", min_value=0.1, max_value=2.5, value=0.2, step=0.1
)

if st.button("Predict"):
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"Predicted Class: {iris.target_names[result[0]]}")
