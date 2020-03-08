import keras 
from keras.models import Model
from keras.layers import Dense,Input,Flatten 
from keras.applications.inception_v3 import InceptionV3 

import os
import numpy as np
import cv2


""" Prepare Data """

INPUT_SHAPE = (256,256)

Data_path = "/home/abhi17/Bacteria Detection/API/Data"

categories = os.listdir(Data_path)

print("\nCategories : ",categories)

train_X = []
train_Y = []

for cat in categories:
    
    label = np.zeros(len(categories))
    label[categories.index(cat)]=1.0
    cat_dir = os.path.join(Data_path,cat)

    for img in os.listdir(cat_dir):

        im = cv2.imread(os.path.join(cat_dir,img))
        im = cv2.resize(im,INPUT_SHAPE)

        im_array = np.array(im)

        train_X.append(im_array)
        train_Y.append(label)


train_X = np.array(train_X)
train_Y = np.array(train_Y)
print(train_X.shape)
print(train_Y.shape)







def Classifier(NUM_CLASSES,input_shape = (256,256,3)):

    base_model = InceptionV3(include_top=False,weights='imagenet',input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable=False

    input_img = Input(input_shape)

    base_out = base_model(input_img)
    
    flat = Flatten()(base_out)
    out =Dense(NUM_CLASSES,activation="softmax")(flat)

    model = Model(inputs=[input_img],output=out)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

classifier = Classifier(4,input_shape=(256,256,3))
classifier.summary()

classifier.fit(train_X,train_Y,epochs=20)
classifier.save('inceptionv3_bacteria.h5')





