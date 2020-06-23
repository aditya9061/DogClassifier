import tensorflow as tf
from base64 import decodebytes
from PIL import Image
import numpy as np
import pickle
import cv2,os
def predict(img_str):
    
    #Settings and loading files and models
    
    height=width=96
    channels=3
    model=tf.keras.models.load_model('classifier_model.h5')
    index_to_classes=open('classes.pkl','rb')
    index_to_classes=pickle.load(index_to_classes)

    
    #image conversion and prediction
    img_str=img_str.encode('utf-8')
    imgdata = decodebytes(img_str)
    fin=open('temp.jpg','wb')
    fin.write(imgdata)

    
    image = np.array(Image.open('temp.jpg'))
    image=cv2.resize(image,(height,width))
    img_array=np.expand_dims(image,axis=0)
    prediction=model.predict(img_array/255)
    
    
    return index_to_classes[int(prediction.argmax(-1))]