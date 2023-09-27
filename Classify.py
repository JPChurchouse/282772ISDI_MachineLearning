import cv2
import numpy as np
import os
import sys
import random
import time
import tqdm
import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model


def ClasifyAll(name_model, path_model, dir_test, dir_output):

  model = load_model(path_model)
  incorrect = 0
  count = 0

  for image in os.listdir(dir_test):
    
    count += 1
    if count % 50 == 0: print(count)
    
    path = os.path.join(dir_test, image)
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    resize = tf.image.resize(img, (256,256))
    res = model.predict(np.expand_dims(resize/255, 0))

    passed = res > 0.5
    colour_back = (0,255,0) if passed else (0,0,255)
    colour_fore = (0,0,0) if passed else (255,255,255)
    msg = "PASS" if passed else "FAIL"

    cv2.rectangle(img, (0,0), (100,40), colour_back, cv2.FILLED)
    cv2.putText(img, msg, (1,30), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_fore, 2)
    
    cv2.imwrite(os.path.join(dir_output,image), img)

    number = int(image.replace(".PNG",""))
    is_pass = number > 400
    if passed != is_pass: incorrect += 1

    print(res)

  print(f"Number of incorrect classifications: {incorrect}")
  return