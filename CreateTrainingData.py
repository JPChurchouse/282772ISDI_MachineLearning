import cv2
import numpy as np
import os
import sys
import random
import time
from tqdm import tqdm
import pickle

# Create a set of training data
def CreateTrainingData(
    dir_raw,
    dir_train,
    img_size = 50,
    name_data = "data",
    name_labels = "labels"
    ):
  
  print("Starting \"CreateTrainingData()\"\n")
  try:

    train_data = []

    categories = getCategories(dir_raw)

    #
    for category in tqdm(categories):

      #
      path = os.path.join(dir_raw,category)
      classid = categories.index(category)

      #
      for image in tqdm(os.listdir(path)):

        #
        grey = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(grey, (img_size,img_size))
        train_data.append([resized/255.0, classid])

    #
    random.shuffle(train_data)

    data = []
    labels = []

    for d, l in train_data:
      data.append(d)
      labels.append(l)

    data = np.array(data).reshape(-1, img_size, img_size, 1)

    #
    path = os.path.join(dir_train, name_data) + ".pickle"
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

    path = os.path.join(dir_train, name_labels) + ".pickle"
    pickle_out = open(path, "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()

  #
  except Exception as exc:
    print("An error occured while creating the data\n" + str(exc))
    return -1

  print("Completed \"CreateTrainingData()\"\n")
  return 0




# Get the categories from the data directory
def getCategories(dir):

  print("Searching for categories")

  categories = []

  # For each thing in the giver directory
  for thing in os.listdir(dir):
    print("Found: %s" %thing)

    # If the thing is a directory, add it as a category
    if os.path.isdir(os.path.join(dir,thing)):
      categories.append(thing)

  print("Category search complete\n")

  # If there are categories, return them
  if categories: return categories

  # Something went wrong, throw exception
  raise Exception("No categories found")