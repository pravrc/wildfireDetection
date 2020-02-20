import streamlit as st
import wildfireDetection.inferenceClassifier as classifier
import json
import numpy as np
from fastai.vision import *

def predict(image):
    with open('configs/inferenceClassifier.json') as config_buffer:    
        config = json.load(config_buffer)
    learn = classifier.load_learner(config['model']['folder'],config['model']['name'])
    # Do predictions for the images
    #classifier.predict(learn,config['input']['images_folder'])
    #return classifier.predict(learn,path)
    return learn.predict(image)
    


st.title("Wildfire Detection")
file_obj = st.file_uploader('Choose an image:', ('jpg', 'jpeg'))
if file_obj is not None:
    st.image(file_obj)
    image = open_image(file_obj)
    #filename = st.text_input('Enter a file path:')
    #print(filename)
    result = predict(image)
    if str(result[0]) == 'smoke':
        st.write('Fire Detected !!!!')
    else:
        st.write('Fire not detected in the image')
    
    probResult = result[2].data.cpu().numpy()
    st.write('Predicted probability of no smoke:', probResult[0])
    st.write('Predicted probability of smoke:', probResult[1])

    

