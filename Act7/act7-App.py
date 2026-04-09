import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
#---------------App Info----------------------------
def appInfo():
    #Add your app information.
    st.title("Object Detection with SSD MobileNet V2")
    st.header("Developed by: Jasmine")
    app_desc = "This app uses a pre-trained deep learning model to detect objects in images. Users can upload an image, and the system will identify and label objects with bounding boxes and confidence scores based on the COCO dataset."

    st.write(app_desc)
#---------------End App Info-------------------------
#---------------Uplode File--------------------
def upload_file():
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        )
    if uploaded_file is not None:

        uploaded_image = Image.open(uploaded_file)

        st.image(uploaded_image, caption="Uploaded Image", width=700)

        return uploaded_file, uploaded_image

    else:

        return None, None
    
#---------------End Uplode File--------------------
#---------------Classify Image---------------------
def classify_image(uploaded_image,model):
    #  Convert image to a format the computer understands (an array)
    x = image.img_to_array(uploaded_image)
    x = np.expand_dims(x, axis=0)  # Add a 'batch' dimension
    x = preprocess_input(x)  # Adjust colors to match what VGG16 expects

    # Make a prediction
    preds = model.predict(x)

    # Convert the math results into human-readable labels
    results = decode_predictions(preds, top=3)[0]

    # Loop through the results to print them cleanly
    for i, (imagenet_id, label, score) in enumerate(results):
        # Score is a decimal (e.g., 0.98), so we multiply by 100 for a percentage
        st.write(f"{i + 1}. {label}: {score * 100:.2f}%")
#---------------End Classify Image---------------------

#Load the model (pre-trained on the ImageNet dataset)
model = VGG16(weights='imagenet')


#Call info 
appInfo()
# Image Upload
st.subheader("Image Upload")
#Call the function uplode file
uploaded_file, uploaded_image=upload_file()

# Resize image to 224x224 pixels, as VGG16 requires



# Show image information

if uploaded_file is not None:
    uploaded_image = uploaded_image.resize((224, 224))
    st.write(f"Image size: {uploaded_image.size} pixels")
    st.write(f"File name: {uploaded_file.name}")
    # Call the classify_image function
    classify_image(uploaded_image, model)
else:
    st.info("No image uploaded yet. Upload an image.")

