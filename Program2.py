# https://www.youtube.com/watch?v=jztwpsIzEGc
# https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb


import cv2
import numpy as np
import os
import sys
import random

import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation

directory_data_training = os.path.join(os.getcwd(),"data")
directory_testimages = os.path.join(os.getcwd(),"images")
directory_output = os.path.join(os.getcwd(),"out")
directory_model = os.path.join(os.getcwd(),"model")

if not os.path.exists(directory_output): os.makedirs(directory_output)
if not os.path.exists(directory_model): os.makedirs(directory_model)

filename_model = "mymodel.keras"
filepath_model = os.path.join(directory_model,"model")

categories = ["pass","fail_a","fail_b"]
img_size = 64

data_training = []


def CreateModel():
    X = []
    y = []

    for categ in categories:
        path = os.path.join(directory_data_training, categ)
        class_id = categories.index(categ)

        for img in os.listdir(path):
            imgarr = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            resarr = cv2.resize(imgarr, (img_size,img_size))
            data_training.append([resarr, class_id])

    random.shuffle(data_training)

    for features,label in data_training:
        X.append(features)
        y.append(label)
    X = np.array(X)

    X = X/255.0

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(len(categories),activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X, np.array(y), batch_size=32, epochs=3, validation_split=0.3)

    # SAVE MODEL
    model.save(filepath_model)

    return



def Clasify():
    mymodel = load_model(filepath_model)
    incorrect = 0
    count = 0

    for file in os.listdir(directory_testimages):
        count += 1
        if count % 50 == 0: print(count)
        
        path = os.path.join(directory_testimages, file)
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        resize = tf.image.resize(img, (256,256))
        res = mymodel.predict(np.expand_dims(resize/255, 0))

        passed = res > 0.5
        colour_back = (0,255,0) if passed else (0,0,255)
        colour_fore = (0,0,0) if passed else (255,255,255)
        msg = "PASS" if passed else "FAIL"

        cv2.rectangle(img, (0,0), (100,40), colour_back, cv2.FILLED)
        cv2.putText(img, msg, (1,30), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_fore, 2)
        
        cv2.imwrite(os.path.join(directory_output,file), img)

        number = int(file.replace(".PNG",""))
        is_pass = number > 400
        if passed != is_pass: incorrect += 1

    print(f"Number of incorrect classifications: {incorrect}")
    return


def main():
    print("Start")
    
    CreateModel()
    Clasify()

    print("End")

    return 0

if __name__ == "__main__":
    sys.exit(main())