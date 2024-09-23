import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# make containers
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Machine Learning Model")
    st.write("This is a machine learning model that predicts")
    st.text("This is a machine learning model that predicts")

with datasets:
    st.header("I like Neural Network")
    st.write("This is a machine learning model that predicts")
    st.text("This is a machine learning model that predicts")
    
    # import data
    # data = pd.read_xlsx("statistics.xlsx")
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head())

    # plot bar chart and others
    st.subheader("Titanic Passengers Count Gender wise")
    st.bar_chart(df['sex'].value_counts())
    
    st.subheader("Titanic Class wise Differentiation")
    st.bar_chart(df['class'].value_counts())
    
    st.subheader("Titanic Age wise Differentiation")
    st.bar_chart(df['age'].value_counts(100))
    st.bar_chart(df['age'].sample(10))
    
with features:
    st.header("Features")
    st.write("This is a machine learning model that predicts")
    st.text("This is a machine learning model that predicts")
    st.markdown('1. ** feature 01: Neural Networks')
    st.markdown('2. ** feature 02: Artificial Intelligence')

with model_training:
    st.header("Artificial Intelligence")
    st.text("This is a machine learning model that predicts")
    st.write("This is a machine learning model that predicts")
    
    # Making Columns 
    input, display = st.columns(2)
    
    # slider to choose value
    max_depth = input.slider("Value selection:", min_value=10, max_value=100, value=20, step=5)
    
# n_estimator
n_estimators = input.selectbox("Select Number of Trees in Random Forest Classifier", options=[50, 100, 200, 300, 'No Limit'])

# Adding list of Features
input.write(df.columns)

# User Input Feature
input_features = input.text_input('What feature would you like to choose')

# Machine Learning Model
model = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
# Condition for No Limit 
if n_estimators == 'No Limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

# Define X and y
X = df[[input_features]]
y = df[['fare']]

# Fit our Model
model.fit(X, y)
pred = model.predict(y)

# Display Metrices
display.subheader("Mean Absolute Error: ")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean Squared Error: ")
display.write(mean_squared_error(y, pred))
display.subheader("R Squared Error: ")
display.write(r2_score(y, pred))