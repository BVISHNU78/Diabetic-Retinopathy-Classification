import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

Model=load_model(r"D:\coding\diabetics_retinopathy_model.h5")
def preprocess_image(image_path,image_size=(128,128)):
    img=tf.keras.utils.load_img(image_path,target_size=image_size)
    img=tf.keras.utils.img_to_array(img)
    img=img/255.0
    img=np.expand_dims(img,axis=0)
    return img
def classify_img(Model,image_path,image_size=(128,128)):
    img=preprocess_image(image_path,image_size)
    prediction=Model.predict(img)
    predicted_class = np.argmax(prediction)
    print(f"prediction probablites:{prediction}")
    print(f"Predicted class: {predicted_class}")
    return predicted_class
image_path=r"C:\Users\Dell\Downloads\eye.jpg"
predicted_class = classify_img(Model, image_path)
print(f"predicted class:{predicted_class}")

