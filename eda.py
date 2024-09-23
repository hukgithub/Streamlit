# Making a DataSceince App on Streamlit where User can load data

import pandas as pd
import numpy as np
import streamlit as st 
import seaborn as sns
from ydata_profiling import ProfileReport #for generating data profiles
from streamlit_pandas_profiling import st_profile_report # for integrating pandas profiling with Streamlit

# Web App Title 
st.markdown ('''
             # EDA Web Application 
            This App is my **Test App** on Streamlit
            ''')

# How to upload a file from PC
with st.sidebar.header('Upload your Dataset (.csv)'):
    uploaded_file = st.sidebar.file_uploader('Upload your file', type=['csv'])
    # Sample Data set of 'titanic
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](df)")
    
# Profiling report for Pandas
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input Data**')
    st.write(df)
    st.write('---')
    st.header('**Profiling  Report**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        @st.cache_data
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                             columns=['A','B','C','D','E'])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
    st.header('**Input Data**')
    st.write(df)
    st.write('---')
    st.header('**Profiling  Report**')
    st_profile_report(pr)