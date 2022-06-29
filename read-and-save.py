# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:04:49 2022

@author: Adhoc
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('VIDEOS/6/Six.MOV')

if cap.isOpened()== False: 
    print('Error opening the video file.')
    
cnt=0

# ret, frame = cap.read()

# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# frame = cv2.resize(frame, (200, 200))

# cv2.imwrite('DATA/0/1' + '.jpg', frame)

# plt.imshow(frame)

while cap.isOpened():
    ret, frame = cap.read()
    
    if(ret == True):
        # print(str(cnt))
        frame = cv2.resize(frame, (200, 200))
        # frame = cv2.resize(frame, (200, 200))
        cv2.imwrite('DATA/6/' + str(cnt) + '.jpg', frame)
        cnt += 1
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
