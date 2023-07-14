# https://www.youtube.com/watch?v=jztwpsIzEGc
# https://github.com/nicknochnack/ImageClassification/blob/main/Getting%20Started.ipynb


import cv2
import numpy as np
import os
import sys
import random

import tensorflow.python as tf

import numpy as np
from matplotlib import pyplot as plt

def getPath(dir, name = "",type = "png"):
    cwd = os.getcwd()
    path = os.path.join(cwd,dir)
    if name != "":
        path = os.path.join(path,f"{name}.{type}")
    return path


def InitData():

    cwd = os.getcwd()
    directory_raw = os.path.join(cwd,"images")
    
    directory_test = os.path.join(cwd,'testing')
    if not os.path.exists(directory_test): os.makedirs(directory_test)
    

    directory_train = os.path.join(cwd,'training')
    directory_fail = os.path.join(directory_train,'fail')
    directory_pass = os.path.join(directory_train,'pass')
    if not os.path.exists(directory_train): os.makedirs(directory_train)
    if not os.path.exists(directory_fail): os.makedirs(directory_fail)
    if not os.path.exists(directory_pass): os.makedirs(directory_pass)
    
    for type in range(3):           # For each type - 0: fail1, 1: fail2, 3: pass
        train = directory_fail if type < 2 else directory_pass

        for block in range (10):    # For each block of 20 images in the 200 images of each type
            rand = random.randint(1,20)

            for index in range (0,20):
                name = type*200 + block*20 + index + 1
                

                path = os.path.join(directory_raw, f"{name}.png")
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                
                dir = train
                if index == rand: 
                    dir = directory_test

                print(f"Processing: {name}")
                    
                dir = os.path.join(dir,f"{name}.png")
                
                cv2.imwrite(dir, img)

    return


def cheat():
    for i in range(8):
        rand = random.randint(1,600)
        print(rand)
        col = (0,255,0) if rand > 400 else (0,0,255)
        msg = "PASS" if rand > 400 else "FAIL"
        
        path = getPath("images",rand)
        assert path is not None, "Image not found"
        print (path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        cv2.rectangle(img, (0,0), (100,40),(0,0,0),cv2.FILLED)        # Background of count
        cv2.putText(img,msg,(1,30), cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
        
        cv2.imshow(path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def runLearning():
    data = tf.keras.utils.image_dataset_from_directory('images')
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    return




def main():
    print("Start")
    #cheat()
    #InitData()
    print("End")

    return 0

if __name__ == "__main__":
    sys.exit(main())