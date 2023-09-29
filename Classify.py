# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 

# EXTERNAL FUNCTIONS

# Use the same function as the training data preparation function
from CreateTrainingData import PrepImage

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
from tensorflow.keras.models import load_model


# MAIN FUNCTION

# cLASSIFY IMAGES
def ClasifyAll(name_model, dir_model, img_size, categories, dir_test, dir_output):

  model = load_model(os.path.join(dir_model,name_model))
  incorrect = 0

  for image in tqdm(os.listdir(dir_test)):
    
    path = os.path.join(dir_test, image)

    id,clas = ClassifyOne(model,categories,path,img_size)

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
    print(f"Number: {number}, Correct: {is_pass==passed}, ID: {id}, Class: {clas}")

    #print(res)

  print(f"Number of incorrect classifications: {incorrect}")
  return


# HELPER FUNCTIONS

# Identify this one image
def ClassifyOne(model, categories, path, img_size):
  image = PrepImage(path, img_size, True)
  prediction = np.round(model.predict(image)[0][0] * (len(categories)-1))
  result = "err"#categories[prediction]
  return prediction,result

