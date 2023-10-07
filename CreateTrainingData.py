# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# This file holds the function to create training data

import Directories as Dir

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
def CreateTrainingData(imsize):
  print("Starting \"CreateTrainingData()\"\n")

  try:

    # SEARCH FOR DATA

    # Init training data
    train_data = []

    # Search for categs
    categories = SearchCategories()

    # Work through each category
    for category in tqdm(categories):

      # Init categ
      path = os.path.join(Dir.categ,category)
      classid = categories.index(category)

      # For each image in this categ
      for image in tqdm(os.listdir(path)):

        # Prepare and include image
        img = PrepImage(os.path.join(path,image), imsize)
        train_data.append([img, classid])


    # PROCESS DATA

    # Shuffle the data
    random.shuffle(train_data)

    # Post process data
    data = []
    labels = []

    for d, l in train_data:
      data.append(d)
      labels.append(l)

    data = np.array(data).reshape(-1, imsize, imsize, 1)


    # EXPORT

    # Export data
    path = os.path.join(Dir.train, Dir.name_data) + f"{imsize}.pickle"
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

    # Export labels
    path = os.path.join(Dir.train, Dir.name_labels) + f"{imsize}.pickle"
    pickle_out = open(path, "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()


  # EXCEPTION HANDLING
  except Exception as exc:
    print("An error occured while creating the data\n" + str(exc))
    raise exc
    return

  # CONCLUDE
  print("Completed \"CreateTrainingData()\"\n")
  return categories


# HELPER FUNCTIONS

# Get the categories from the data directory
def SearchCategories():
  print("Searching for categories")

  categories = []

  # For each thing in the given directory
  for thing in tqdm(os.listdir(Dir.categ)):
    print(f"Found: {thing}")

    # If the thing is a directory, add it as a category
    if os.path.isdir(os.path.join(Dir.categ,thing)):
      categories.append(thing)

  print("Category search complete\n")

  # If there are categories, return them
  if categories:

    path = os.path.join(os.getcwd(), f"{Dir.name_categs}.pickle")
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