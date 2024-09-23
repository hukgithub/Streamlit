import streamlit as st 
from PIL import Image

st.write("""
         # Add Media files in Streamlit Web Application
         """)

# Adding Image
st.write("""
         ## Snow Leopard
         """)
image_1 = Image.open("image.jpg")
st.image(image_1)

# Adding Video
st.write("""
         ## Video
         """)
video_1 = open("video.mp4", "rb")
st.video(video_1)

# Adding Audio
st.write("""
         ## Audio
         """)
Audio_1 = open("audio.mp3", "rb")
st.audio(Audio_1)