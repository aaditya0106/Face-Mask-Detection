# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:36:36 2020

@author: Aaditya Chopra
"""


from os import listdir

# making training and testing outputs
train_img_without_mask = "Dataset/train/without_mask"
train_img_with_mask = "Dataset/train/with_mask"
test_img_without_mask = "Dataset/test/without_mask"
test_img_with_mask = "Dataset/test/with_mask"

y_train = list()
y_test = list()

for i in range(0,len(listdir(train_img_with_mask))):
    y_train.append(1)

for i in range(0,len(listdir(train_img_without_mask))):
    y_train.append(0)
    
for i in range(0,len(listdir(test_img_with_mask))):
    y_test.append(1)

for i in range(0,len(listdir(test_img_without_mask))):
    y_test.append(0)
    
    

#working on x_train
from keras.preprocessing.image import img_to_array 
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np

x_train = []
x_test = []

for name in listdir(train_img_with_mask):
    filename = train_img_with_mask + '/' + name
    img = cv2.imread(filename)
    img = cv2.resize(img, (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    x_train.append(img)

for name in listdir(train_img_without_mask):
    filename = train_img_without_mask + '/' + name
    img = cv2.imread(filename)
    img = cv2.resize(img, (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    x_train.append(img)
    
for name in listdir(test_img_with_mask):
    filename = train_img_with_mask + '/' + name
    img = cv2.imread(filename)
    img = cv2.resize(img, (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    x_test.append(img)
    
for name in listdir(test_img_without_mask):
    filename = test_img_without_mask + '/' + name
    img = cv2.imread(filename)
    img = cv2.resize(img, (224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    x_test.append(img)

x_train = np.array(x_train, dtype="float32")
x_test = np.array(x_test, dtype="float32")
y_train = np.array(y_train)
y_test = np.array(y_test)

#working on model now
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model

def create_model():
    basemodel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    
    mainmodel = basemodel.output
    mainmodel = AveragePooling2D(pool_size=(7,7))(mainmodel)
    mainmodel = Flatten()(mainmodel)
    mainmodel = Dense(128, activation='relu')(mainmodel)
    mainmodel = Dropout(0.5)(mainmodel)
    mainmodel = Dense(1,activation='sigmoid')(mainmodel)
    
    model = Model(inputs = basemodel.input, outputs = mainmodel)
    
    for layer in basemodel.layers:
        layer.trainable = False
    
    return model
    
model = create_model()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 32
Epoch = 20

#define checkpoint callback
from keras.callbacks import ModelCheckpoint

checkpoint_filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, verbose=1, batch_size=32, callbacks=[checkpoint])
