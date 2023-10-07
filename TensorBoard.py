# INFO

# Jamie Churchouse, 20007137
# Massey University, Palmerston North, NZ
# 282772 Industrial Systems Design and Integration
# Machine Learning Project, 2023-10-06 1800
# 
# This file contains custom TensorBoard functionality


# LIBRARIES

import Directories as Dir

import webbrowser as wb
import subprocess as sp


# FUNCTIONS

# TensorBoard launch function
def Launch():

  print("Launching TensorBoard")

  cmd = "(cd %s && tensorboard --logdir=%s/)" % (Dir.tboard, Dir.tboard)
  sp.Popen(cmd, shell = True)

  url = "http://localhost:6006/?darkMode=true"
  wb.open(url)

  print("TensorBoard launch complete\n")
  return