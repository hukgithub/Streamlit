import streamlit as st
import seaborn as sns

st.header("DATA SCIENCES")
st.text("Machine Learning")
st.text("Deep Learning + Neural Networks")
st.header("ARTIFICIAL INTELLIGENCE")

df = sns.load_dataset('iris')
st.write(df[['species', 'sepal_length', 'petal_length']].head(10))

st.bar_chart(df['sepal_length'])
st.line_chart(df['sepal_length'])


