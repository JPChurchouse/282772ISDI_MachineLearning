import cv2
import numpy as np
import os
import sys
import random

import tensorflow.python as tf



def getImgPath(dir, name):
    cwd = os.getcwd()
    path = os.path.join(cwd,dir,f"{name}.png")
    return path


def cheat():
    for i in range(8):
        rand = random.randint(1,600)
        print(rand)
        col = (0,255,0) if rand > 400 else (0,0,255)
        msg = "PASS" if rand > 400 else "FAIL"
        
        path = getImgPath("images",rand)
        assert path is not None, "Image not found"
        print (path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        cv2.rectangle(img, (0,0), (100,40),(0,0,0),cv2.FILLED)        # Background of count
        cv2.putText(img,msg,(1,30), cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
        
        cv2.imshow(path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def runLearning():
    img = tf.image.loa



def main():
    print("Start")
    cheat()
    print("End")

    return 0

if __name__ == "__main__":
    sys.exit(main())