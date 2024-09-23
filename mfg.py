import streamlit as st 
from streamlit_embedcode import github_gist

from streamlit import caching

link = ""

st.write("Embed Github Gist:")
github_gist(link)

