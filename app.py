
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.model.load_model('./alzheimer_model.keras')
    return model

model = load_model()
st.write("""
        # Alzheimer's Disease Detection
        """)

file = st.file.uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model): 
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    print("This image belongs to {} with a {:2f} percent confidence".format(class_names[np.argmax(score)], 100 * np.max(score)))
