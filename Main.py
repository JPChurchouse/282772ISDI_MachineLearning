# https://www.youtube.com/watch?v=jztwpsIzEGc
# https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb

# IMPORT

# Functions
import CreateTrainingData as Create
import Classify as Classify
import TrainModel as Train

# Libraries
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
dir_raw     = os.path.join(os.getcwd(),"raw_data")
dir_train   = os.path.join(os.getcwd(),"training_data")
dir_model   = os.path.join(os.getcwd(),"models")
dir_tboard  = os.path.join(os.getcwd(),"tensor_board")
dir_test    = os.path.join(os.getcwd(),"master_images")
dir_output  = os.path.join(os.getcwd(),"classified_images")

dirs_cre = [dir_model,dir_train,dir_tboard,dir_output]
dirs_req = [dir_raw,dir_test]

def CheckDirectories(clear = False):

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




# MAIN
def main():

  print("Starting MAIN\n")
  ret = 0

  CheckDirectories()

  #ret = Create.CreateTrainingData(dir_raw,dir_train,img_size)
  if not ret == 0 : return ret

  ret = Train.BuildModel(dir_train,dir_model,dir_tboard,name_model)
  if not ret == 0 : return ret

  LaunchTensorBoard(dir_tboard)

  print("Concluding MAIN")

  return 0

print("\n\n")
if __name__ == "__main__":
  sys.exit(main())