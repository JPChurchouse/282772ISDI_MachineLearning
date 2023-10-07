# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
#
# This file contains functionality to classify the images


#█▄─▀█▄─▄█─▄▄─█─▄─▄─███▄─▄█▄─▀█▀─▄█▄─▄▄─█▄─▄███▄─▄▄─█▄─▀█▀─▄█▄─▄▄─█▄─▀█▄─▄█─▄─▄─█▄─▄▄─█▄─▄▄▀█
#██─█▄▀─██─██─███─██████─███─█▄█─███─▄▄▄██─██▀██─▄█▀██─█▄█─███─▄█▀██─█▄▀─████─████─▄█▀██─██─█
#▀▄▄▄▀▀▄▄▀▄▄▄▄▀▀▄▄▄▀▀▀▀▄▄▄▀▄▄▄▀▄▄▄▀▄▄▄▀▀▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▀▄▄▄▀▄▄▄▄▄▀▄▄▄▀▀▄▄▀▀▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▀▀


# LIBRARIES

# External functions
# Use the same function as the training data preparation function
from CreateTrainingData import PrepImage

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
from tensorflow.keras.models import load_model


# MAIN FUNCTION

# Classify images
def ClasifyAll(name_model, dir_model, img_size, categories, dir_test, dir_output):
  raise NotImplementedError

  model = load_model(os.path.join(dir_model,name_model))
  incorrect = 0

  for image in tqdm(os.listdir(dir_test)):
    
    path = os.path.join(dir_test, image)

    categ,id = ClassifyOne(model,categories,path,img_size)

    passed = id == 0
    #colour_back = (0,255,0) if passed else (0,0,255)
    #colour_fore = (0,0,0) if passed else (255,255,255)
    #msg = "PASS" if passed else "FAIL"

    #cv2.rectangle(img, (0,0), (100,40), colour_back, cv2.FILLED)
    #cv2.putText(img, msg, (1,30), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_fore, 2)
    
    #cv2.imwrite(os.path.join(dir_output,image), img)

    number = int(image.replace(".PNG",""))
    is_pass = number > 400
    if passed != is_pass: incorrect += 1
    print(f"Number: {number}, Correct: {is_pass==passed}, ID: {id}, Class: {categ}")

    #print(res)

  print(f"Number of incorrect classifications: {incorrect}")
  return


# HELPER FUNCTIONS

# Identify this one image
def ClassifyOne(model, categories, path, img_size):
  raise NotImplementedError
  image = PrepImage(path, img_size, True)
  index = np.argmax(model.predict(image))[0]
  categ = categories[index]
  return categ,index