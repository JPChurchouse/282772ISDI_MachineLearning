# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# This file holds the function to train the model


# LIBRARIES
import cv2
import numpy as np
import os
import sys
import random
import time
from tqdm import tqdm
import pickle

import tensorflow as tf
from tensorflow.keras.metrics   import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras.layers  import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras.callbacks import TensorBoard



# Create the model
def BuildModel(
    dir_train,
    dir_model,
    dir_tboard,
    name_model,
    name_data = "data",
    name_labels = "labels"
    ):

  print("Starting \"BuildModel()\"")
  try:

    timenow = int(time.time() * 1000)
    name_this = name_model + "_" + str(timenow)
    print("This model's name is \"%s\"\n" %name_this)

    path = os.path.join(dir_tboard, name_this)
    tensorboard = TensorBoard(log_dir = path)

    #

    path = os.path.join(dir_train, "%s.pickle" % name_data)
    pickle_in = open(path, "rb")
    data = np.array(pickle.load(pickle_in))
    pickle_in.close()

    path = os.path.join(dir_train, "%s.pickle" % name_labels)
    pickle_in = open(path, "rb")
    labels = np.array(pickle.load(pickle_in))
    pickle_in.close()


    num_categories = len(set(labels))
    print("Number of categories detected: %d"%num_categories)

    #
    model =  Sequential()

    model.add(Conv2D(64, (3,3), input_shape = data.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(num_categories))
    model.add(Activation("sigmoid"))

    model.compile(
      loss = "categorical_crossentropy",
      optimizer = "adam",
      metrics = ["accuracy"]
      )

    model.fit(
      data,
      labels,
      batch_size = 30,
      validation_split = 0.1,
      epochs = 20,
      callbacks = [tensorboard]
      )
    
    path = os.path.join(dir_model, name_this)
    model.save(path)

  #
  except Exception as exc:
    print("An error occured while creating the model" + "\n" + str(exc))
    raise exc
    return

  print("Concluded \"BuildModel()\"\n")
  return name_this