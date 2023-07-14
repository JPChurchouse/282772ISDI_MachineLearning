import cv2
import numpy as np
import os
import sys
import random

import tensorflow as tf


def getPath(dir, name):
    cwd = os.getcwd()
    path = os.path.join(cwd,dir,name)
    return path


def DoStuff():
    for i in range(8):
        rand = random.randint(1,600)
        print(rand)
        col = (0,255,0) if rand > 400 else (0,0,255)
        msg = "PASS" if rand > 400 else "FAIL"
        
        path = getPath("images",f"{rand}.png")
        assert path is not None, "Image not found"
        print (path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        cv2.rectangle(img, (0,0), (100,40),(0,0,0),cv2.FILLED)        # Background of count
        cv2.putText(img,msg,(1,30), cv2.FONT_HERSHEY_SIMPLEX,1,col,2)
        
        cv2.imshow(path, img)

def main():
    print("Start")
    DoStuff()
    print("End")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())