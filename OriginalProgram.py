# INFO

# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
#
# This file is the original program that categorises the images

# Notes
# https://www.youtube.com/watch?v=jztwpsIzEGc
# https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb


# LIBRARIES

# External functions
import Directories as Dir

# Libraries
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

# Tensorflow
# Ignore the yellow squiggles - it's a known bug
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras.layers  import Conv2D, MaxPooling2D, Dense, Flatten


# DIRECTORIES, NAMES, PATHS
# name : name of a file, does not include the path to it
# dire : directory, does not specify an individual file
# path : full path to a specific file

name_model   = "classifier"
path_model   = os.path.join(Dir.model,name_model)
dire_train   = os.path.join(os.getcwd(), f"{name_model}_training")

# If the training data does not already exist, create it and the pass fail folders
if not os.path.exists(dire_train):
  print("Making training data")

  # Create necessary directories
  os.makedirs(dire_train)

  pas = os.path.join(dire_train,"PASS")
  fai = os.path.join(dire_train,"FAIL")
  os.makedirs(pas)
  os.makedirs(fai)

  # For each image
  for file in tqdm(os.listdir(Dir.raw)):

    # Categorise
    number   = int(file.replace(".PNG",""))
    categ    = pas if number > 400 else fai

    # Paths
    readfrom = os.path.join(Dir.raw,file)
    writeto  = os.path.join(categ,file)

    # Make operation
    cv2.imwrite(writeto, cv2.imread(readfrom, cv2.IMREAD_COLOR))

  print("Training data complete\n")


# FUNCTIONS

# Create model function
def CreateModel():
  print("Creating model...")

  # INITALISE DATA

  # Training directory
  data       = tf.keras.utils.image_dataset_from_directory(dire_train)
  iterator   = data.as_numpy_iterator()

  # Set batches and training sizes
  batch      = iterator.next()
  data       = data.map(lambda x,y: (x/255, y))
  data.as_numpy_iterator().next()

  train_size = int(len(data)*.7)
  val_size   = int(len(data)*.2) +1
  test_size  = int(len(data)*.1) +1

  train      = data.take(train_size)
  val        = data.skip(train_size).take(val_size)
  test       = data.skip(train_size+val_size).take(test_size)


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

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = Dir.tboard)
  hist = model.fit(train, epochs = 20, validation_data = val, callbacks = [tensorboard_callback])


  # MODEL STATS

  pre = Precision()
  re  = Recall()
  acc = BinaryAccuracy()

  for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

  print(pre.result(), re.result(), acc.result())


  # SAVE MODEL

  model.save(path_model)


  print(f"Model created\n{path_model}\n")
  return


# Classify all images
def Classify():
  print(f"Classifying images\n{path_model}\n")

  # Load the trained model
  mymodel      = load_model(path_model)
  incorrect    = 0

  # For each image in the raw directory
  for file in tqdm(os.listdir(Dir.raw)):

    # Init, resize
    path       = os.path.join(Dir.raw, file)
    image      = cv2.imread(path, cv2.IMREAD_COLOR)
    resize     = tf.image.resize(image, (256,256))

    # Predict
    res = mymodel.predict(np.expand_dims(resize/255, 0))

    # Export image
    passed     = res > 0.5
    col_bak    = (  0,255,  0) if passed else (  0,  0,255)
    col_fro    = (  0,  0,  0) if passed else (255,255,255)
    message    = "PASS"        if passed else "FAIL"

    cv2.rectangle(image, (  0,  0), (100, 40), col_bak, cv2.FILLED)
    cv2.putText  (image,   message, (  1, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col_fro, 2)

    cv2.imwrite(os.path.join(Dir.output, file), image)

    # Evaluate model's result
    number     = int(file.replace(".PNG",""))
    is_pass    = number > 400
    if passed != is_pass: incorrect += 1

  # Conclude
  print(f"Number of incorrect classifications: {incorrect}")
  print(f"Output directory:\n{Dir.output}")
  print( "Classification complete\n")
  return


# MAIN

# Main function
def main():
  print("Start")

  # Verify directories
  Dir.Verify()

  # Create the model
  CreateModel()

  # Classify images
  Classify()

  print("End")
  return 0

if __name__ == "__main__":
  sys.exit(main())