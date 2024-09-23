# Animated Plot using Plotly on Streamlit

import streamlit as st 
import pandas as pd
import plotly.express as px

st.title("Plotly Express Example on Streamlit")
# Data Set 
df = px.data.gapminder()
st.write(df)
st.write(df.columns)

# Summary Stat
st.write(df.describe())

# Data Management
year_option = df['year'].unique().tolist()
year = st.selectbox("Which Year would you like to Plot", year_option, 0)
# df = df[df['year'] == year]

# Plotting a Scatter Plot
fig = px.scatter(df, x = 'gdpPercap', y = 'lifeExp', size = 'pop', color = 'country', hover_name = 'country',
                 log_x=True, size_max=55, range_x=[100, 100000], range_y=[20, 90],
                 animation_frame='year', animation_group='country')

fig.update_layout(width=800, height=400)

st.write(fig)
