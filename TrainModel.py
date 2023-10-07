# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# This file holds the function to train the model


# LIBRARIES

# External functions
import Directories as Dir

# Libraries
import cv2
import numpy as np
import os
import sys
import random
import time
from tqdm import tqdm
import pickle

# Tensorflow
import tensorflow as tf
from tensorflow.keras.metrics   import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras.layers  import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical


# FUNCTIONS

# Create the model
def BuildModel(params):
  print("Starting \"BuildModel()\"")

  # Import parameters
  param_name   = params [ "name" ]
  param_epochs = params ["epochs"]
  param_layers = params ["layers"]
  param_neuros = params ["neuros"]
  param_imsize = params ["imsize"]

  try:

    # INITALISE

    # Update model name to include the time
    timenow     = int(time.time() * 1000)
    param_name  = param_name + "_" + str(timenow)
    print(f"This model's name is \"{param_name}\"\n")

    # Init Tensorboard
    path        = os.path.join(Dir.tboard, param_name)
    tensorboard = TensorBoard(log_dir = path)

    # Import training Data
    path        = os.path.join(Dir.train, f"{Dir.name_data}{param_imsize}.pickle")
    pickle_in   = open(path, "rb")
    data        = np.array(pickle.load(pickle_in))
    pickle_in.close()

    # Import training Labels
    path        = os.path.join(Dir.train, f"{Dir.name_labels}{param_imsize}.pickle")
    pickle_in   = open(path, "rb")
    labels      = np.array(pickle.load(pickle_in))
    pickle_in.close()

    # Calculate categories
    num_categs  = len(set(labels))
    print(f"Number of categories detected: {num_categs}")

    #data   = to_categorical(data,num_categs)
    #labels = to_categorical(labels,num_categs)


    # BUILD MODEL

    model = Sequential()

    # First layer
    shape = (param_imsize,param_imsize,1)
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape = shape))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())

    # Intermediate layer(s) - decided by the input params
    for i in range(1,param_layers):
      model.add(Dense(param_neuros, activation='relu'))

    # Final layer
    model.add(Flatten())
    model.add(Dense(param_neuros*2, activation='relu'))

    # Output layer
    model.add(Dense(num_categs,activation="softmax"))

    # Build
    model.compile(
      loss = "sparse_categorical_crossentropy",
      optimizer = "adam",
      metrics = ["accuracy"]
      )

    model.fit(
      data,
      labels,
      batch_size = 30,
      validation_split = 0.2,
      epochs = param_epochs,
      callbacks = [tensorboard]
      )

    # Save model
    path = os.path.join(Dir.model, param_name)
    model.save(path)


  # EXCEPTION HANDLING
  except Exception as exc:
    print("An error occured while creating the model" + "\n" + str(exc))
    print(param_name)
    raise exc
    return


  # CONCLUDE
  print("Concluded \"BuildModel()\"\n")
  return param_name