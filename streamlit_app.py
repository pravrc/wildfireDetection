import streamlit as st

st.title("Wildfire Detection")
file_obj = st.file_uploader('Choose an image:', ('jpg', 'jpeg'))
if file_obj is not None:
    st.image(file_obj)
st.write("No fire!")

