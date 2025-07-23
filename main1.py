import streamlit as st
import keras
import tensorflow as tf 
from PIL import Image
import numpy as np



Class_names= ["Early_blight" , "Late_blight" ,"Healthy" ]


# Load your saved model
model = keras.models.load_model("PlantVillagePotataoCLassifier.h5")


st.title("Poatato-Plant-Disease-Classifier")


label = "Upload Your Potatao Plant Image"
uploaded_file = st.file_uploader(label, type=["jpg", "jpeg", "png"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")



if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    # Assuming `image` is a PIL image (e.g. from st.file_uploader)
    image = image.resize((224, 224))  # Resize to match your model's input size
    image = keras.preprocessing.image.img_to_array(image)       # Convert to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension: (1, 256, 256, 3)
    

    # Now make prediction
    prediction = model.predict(image)
    predicted_class = Class_names[np.argmax(prediction[0])]
    st.subheader(f"Your Potato Plant is {predicted_class}")
    Cf = round(np.max(prediction[0] * 100) , 2)
    st.write(f"Confidence Score {Cf}%")
    
  

        


