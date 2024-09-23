import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the Iris dataset and train a logistic regression model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Streamlit app
st.title("Iris Flower Prediction App")
st.write("Enter the parameters to predict the type of Iris flower.")

# Input text boxes for features
sepal_length = st.number_input("Sepal Length (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), float(iris.data[:, 0].mean()))
sepal_width = st.number_input("Sepal Width (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), float(iris.data[:, 1].mean()))
petal_length = st.number_input("Petal Length (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), float(iris.data[:, 2].mean()))
petal_width = st.number_input("Petal Width (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), float(iris.data[:, 3].mean()))

# Prepare input for the model
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Make prediction
prediction = model.predict(input_features)
prediction_proba = model.predict_proba(input_features)

# Map the prediction to the Iris species
iris_species = iris.target_names[prediction][0]

# Display the prediction
st.subheader("Prediction")
st.write(f"The predicted Iris species is: **{iris_species}**")

st.subheader("Prediction Probability")
st.write(f"Setosa: {prediction_proba[0][0]:.2f}, Versicolor: {prediction_proba[0][1]:.2f}, Virginica: {prediction_proba[0][2]:.2f}")