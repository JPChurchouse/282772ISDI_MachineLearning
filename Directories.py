# INFO
# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# This file handles creating, verifying, and handling directories


# LIBRARIES
import os


# DIRECTORIES & FILEPATHS
cwd    = os.getcwd()

raw    = os.path.join(cwd,"master_images")
categ  = os.path.join(cwd,"categorised")

train  = os.path.join(cwd,"training_data")
model  = os.path.join(cwd,"models")
tboard = os.path.join(cwd,"tensorboard")
output = os.path.join(cwd,"classified_images")

# Lists of required directories
dirs_required  = [categ,raw]
dirs_to_create = [model,train,tboard,output]



# FUNCTIONS

# Function to verify directory integrities
def Verify():

  fail = False

  for dir in dirs_to_create:
    if not os.path.exists(dir):
      os.makedirs(dir)
      print("Created output directory:\n%s\n" %dir)
  
  for dir in dirs_required:
    if not os.path.exists(dir):
      print("Missing input directory:\n%s\n" %dir)
      fail = True

  if fail: raise Exception("Unable to continue - one or more required input directories are missing")

  return