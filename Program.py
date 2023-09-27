# https://www.youtube.com/watch?v=jztwpsIzEGc
# https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb

# IMPORT
import cv2
import numpy as np
import os
import sys
import random
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.metrics   import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras.layers  import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras.callbacks import TensorBoard



# DIRECTORIES & FILEPATHS
directory_trainingdata = os.path.join(os.getcwd(),"data")
directory_testimages = os.path.join(os.getcwd(),"images")
directory_output = os.path.join(os.getcwd(),"out")
directory_model = os.path.join(os.getcwd(),"model")
directory_log = os.path.join(os.getcwd(),"log")

if not os.path.exists(directory_output) : os.makedirs(directory_output)
if not os.path.exists(directory_model)  : os.makedirs(directory_model)
if not os.path.exists(directory_log)  : os.makedirs(directory_log)

name = "name"
filename_model = "%d_%s" %(time.time(),name)
filepath_model = os.path.join(directory_model,"model")

tensorboard_callback = TensorBoard(log_dir = directory_log + "/thing")

categories = ["pass", "fail_a", "fail_b"]


# FUNCTIONS
size = 50

# Create the model
def CreateModel():
  vector_data = trainingdata[0].reshape(-1, size, size, 1)
  vector_labels = trainingdata[1]

  model =  Sequential()

  model.add(Conv2D(64, (3,3), input_shape = vector_data.shape[1:]))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(64, (3,3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation("relu"))

  model.add(Dense(2))
  model.add(Activation("sigmoid"))

  model.compile(loss = "categorical", optimizer = "adam", metrics = ["accuracy"])

  model.fit(vector_data, 
            vector_labels, 
            batch_size = 30, 
            validation_split = 0.1, 
            epochs = 20,
            callbacks = [tensorboard_callback])
  
  return


trainingdata = []


def CreateTrainingData():
  
  for category in categories:

    path = os.path.join(directory_trainingdata,category)
    classid = categories.index(category)

    for image in tqdm(os.listdir(path)):

      grey = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)
      resized = cv2.resize(grey, (size,size))
      trainingdata.append([resized/255.0, classid])

  random.shuffle(trainingdata)
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

    print(res)

  print(f"Number of incorrect classifications: {incorrect}")
  return



# MAIN
def main():
  print("Start")
  
  CreateTrainingData()
  CreateModel()
  #Clasify()

  print("End")

  return 0

if __name__ == "__main__":
  sys.exit(main())