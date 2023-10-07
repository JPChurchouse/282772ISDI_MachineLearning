# INFO

# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# adsfadsfasfd


# Notes
# https://www.youtube.com/watch?v=jztwpsIzEGc
# https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb


# EXTERNAL FUNCTIONS

import CreateTrainingData as Create
import Classify as Classify # wow! original!
import TrainModel as Train
import Directories as Dir
import TensorBoard as TB


# LIBRARIES

import numpy as np
import os
import sys
import pickle
from tqdm import tqdm


# TESTING PARAMETERS

# Test parameters
test_layers = [  2,  3,  4]
test_imsize = [ 32, 64,128]
test_neuros = [ 16, 32, 64]
test_epochs = [ 10, 15, 20]

# Generate name from parameters
def GenName(L,I,N,E):
  return f"L{L}I{I}N{N}E{E}"

# Run all tests
def RunTests():

  model_names = []
  params = {}

  # Verify directory integrity
  Dir.Verify()

  # Create the training data
  #for size in tqdm(test_imsize) : Create.CreateTrainingData(size)

  # Launch Tensorboard
  TB.Launch()

  # Run through each parameter
  for L in test_layers:
    for I in test_imsize:
      for N in test_neuros:
        for E in test_epochs:

          name = GenName (L,I,N,E)
          params ["layers"] = L
          params ["imsize"] = I
          params ["neuros"] = N
          params ["epochs"] = E
          params [ "name" ] = name

          print(f"\nNow processing: {name}\n")

          model_names.append(Train.BuildModel(params))

          print(f"\nCompleted processing: {name}\n")

  # Conclude
  print(f"Testing complete\nModel names are as follows:")
  for name in model_names: print(name)
  print(f"Directory:\n{Dir.model}")
  print("\nDONE")



# FUNCTIONS

# Function to import category names from a file
def ImportCategories(name_categs = "categs"):
  try:
    path = os.path.join(os.getcwd(),f"{name_categs}.pickle")
    pickle_in = open(path, "rb")
    categories = np.array(pickle.load(pickle_in))
    pickle_in.close()
    return categories
  except:
    return ["A","B","C","D"]


# MAIN
def main():

  print("Starting MAIN\n")

  # Run sequential testing
  RunTests()

  # Verify directories
  #Dir.Verify()

  # Launch TensorBoard
  #TB.Launch(Dir.tboard)

  # Create training data
  #Create.CreateTrainingData(params)

  # Build and train model
  #name_model = Train.BuildModel(params)

  # Classify images with model // NOT OPERATIONAL //
  #Classify.ClasifyAll(params)

  print("Concluding MAIN\n")

  return 0

# OS functionality
print("\n\n")
if __name__ == "__main__":
  sys.exit(main())