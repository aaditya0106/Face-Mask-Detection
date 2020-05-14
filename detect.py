# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:13:10 2020

@author: Aaditya Chopra
"""


from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2

def prediction(test_img, model):
    
    test_img = cv2.resize(test_img, (224,224))
    test_img = img_to_array(test_img)
    test_img = preprocess_input(test_img)
    test_array = []
    test_array.append(test_img)
    test_array = np.array(test_array)
    
    test_result = model.predict(test_array)
    
    return test_result
        

model_weights = "model_weights.h5"
model = load_model(model_weights)

video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    
    result = prediction(img, model)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50,50)
    fontscale = 1
    thickness = 1
    
    if result > 0.5:
        color = (0,255,0)
        img = cv2.putText(img, 'Mask Detected', org, font, fontscale, color, thickness) 
    
    else :
        color = (0,0,255)
        img = cv2.putText(img, 'Mask Not Detected', org, font, fontscale, color, thickness) 
    
    cv2.imshow('Video', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
    
video_capture.release()
cv2.destroyAllWindows()