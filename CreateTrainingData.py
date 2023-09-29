# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# This file holds the function to create training data


# LIBRARIES
import cv2
import numpy as np
import os
import sys
import random
import time
from tqdm import tqdm
import pickle


# MAIN FUNCTION

# Create a set of training data
def CreateTrainingData(
    dir_raw,
    dir_train,
    img_size = 50,
    name_data = "data",
    name_labels = "labels",
    name_categs = "categs"
    ):
  
  print("Starting \"CreateTrainingData()\"\n")
  try:

    train_data = []

    categories = SearchCategories(dir_raw,name_categs)

    #
    for category in tqdm(categories):

      #
      path = os.path.join(dir_raw,category)
      classid = categories.index(category)

      #
      for image in tqdm(os.listdir(path)):

        #
        img = PrepImage(os.path.join(path,image), img_size)
        train_data.append([img, classid])

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
    raise exc
    return

  print("Completed \"CreateTrainingData()\"\n")
  return categories


# HELPER FUNCTIONS

# Get the categories from the data directory
def SearchCategories(dir, name_categs = "categs"):

  print("Searching for categories")

  categories = []

  # For each thing in the given directory
  for thing in tqdm(os.listdir(dir)):
    print("Found: %s" %thing)

    # If the thing is a directory, add it as a category
    if os.path.isdir(os.path.join(dir,thing)):
      categories.append(thing)

  print("Category search complete\n")

  # If there are categories, return them
  if categories:

    path = os.path.join(os.getcwd(), "%s.pickle" % name_categs)
    pickle_out = open(path, "wb")
    pickle.dump(categories, pickle_out)
    pickle_out.close()

    return categories

  # Something went wrong, throw exception
  raise Exception("No categories found")


# Prepare image
def PrepImage(path, img_size, reshape = False):
  out = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  out = cv2.resize(out, (img_size,img_size))
  out = out / 255.0
  # Don't know why I can't reshape the training data here, but the classification is happy 🤷‍♂️
  if reshape: out = out.reshape(-1, img_size, img_size, 1)
  return out