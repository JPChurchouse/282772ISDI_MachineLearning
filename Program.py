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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

directory_trainingdata = os.path.join(os.getcwd(),"data")
directory_testimages = os.path.join(os.getcwd(),"images")
directory_output = os.path.join(os.getcwd(),"out")
directory_model = os.path.join(os.getcwd(),"model")

if not os.path.exists(directory_output): os.makedirs(directory_output)
if not os.path.exists(directory_model): os.makedirs(directory_model)

filename_model = "mymodel.keras"
filepath_model = os.path.join(directory_model,"model")


def CreateModel():

    # INITALISE DATA
    data = tf.keras.utils.image_dataset_from_directory(directory_trainingdata)
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    data = data.map(lambda x,y: (x/255, y))
    data.as_numpy_iterator().next()
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)+1
    test_size = int(len(data)*.1)+1
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    # CREATE MODEL
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    # FIT MODEL
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=directory_model)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    # MODEL STATS
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(pre.result(), re.result(), acc.result())

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
    
    #CreateModel()
    Clasify()

    print("End")

    return 0

if __name__ == "__main__":
    sys.exit(main())