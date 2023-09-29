# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# adsfadsfasfd


# NOTES
# https://www.youtube.com/watch?v=jztwpsIzEGc
# https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb


# EXTERNAL FUNCTIONS
import CreateTrainingData as Create
import Classify as Classify
import TrainModel as Train


# LIBRARIES
import cv2
import numpy as np
import os
import sys
import random
import time
from tqdm import tqdm
import pickle
import webbrowser
import subprocess


# DIRECTORIES & FILEPATHS
dir_cwd     = os.getcwd()
dir_raw     = os.path.join(dir_cwd,"raw_data")
dir_train   = os.path.join(dir_cwd,"training_data")
dir_model   = os.path.join(dir_cwd,"models")
dir_tboard  = os.path.join(dir_cwd,"tensor_board")
dir_test    = os.path.join(dir_cwd,"master_images")
dir_output  = os.path.join(dir_cwd,"classified_images")

dirs_cre = [dir_model,dir_train,dir_tboard,dir_output]
dirs_req = [dir_raw,dir_test]

# Function to verify directory integrities
def CheckDirectories():

  fail = False

  for dir in dirs_cre:
    if not os.path.exists(dir):
      os.makedirs(dir)
      print("Created output directory:\n%s\n" %dir)
  
  for dir in dirs_req:
    if not os.path.exists(dir):
      print("Missing input directory:\n%s\n" %dir)
      fail = True

  if fail: raise Exception("Unable to continue - one or more required input directories are missing")

  return

name_model = "namething"

img_size = 50


# TensorBoard launch function
def LaunchTensorBoard(dir_tboard):

  print("Launching TensorBoard")

  cmd = "(cd %s && tensorboard --logdir=%s/)" % (dir_tboard, dir_tboard)
  subprocess.Popen(cmd, shell = True)

  url = "http://localhost:6006/?darkMode=true"
  webbrowser.open(url)

  print("TensorBoard launch complete\n")
  return



def ImportCategories(name_categs = "categs"):
  try:
    path = os.path.join(os.getcwd(),"%s.pickle" % name_categs)
    pickle_in = open(path, "rb")
    categories = np.array(pickle.load(pickle_in))
    pickle_in.close()
    return categories
  except:
    return ["A","B","C","D"]




# MAIN
def main():

  print("Starting MAIN\n")

  CheckDirectories()

  categories = ImportCategories()
  #LaunchTensorBoard(dir_tboard)
  #Create.CreateTrainingData(dir_raw,dir_train,img_size)
  name_model = "asdf"
  name_model = Train.BuildModel(dir_train,dir_model,dir_tboard,img_size,name_model)



  Classify.ClasifyAll(name_model,dir_model,img_size,categories,dir_test,dir_output)

  print("Concluding MAIN")

  return 0

print("\n\n")
if __name__ == "__main__":
  sys.exit(main())



