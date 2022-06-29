# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:51:05 2022

@author: Adhoc
"""

from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model=tf.keras.models.load_model('luck-two')


data_gen = ImageDataGenerator()
classes=data_gen.flow_from_directory(directory="DATA/",
                                      batch_size=35).class_indices

results = list(classes.keys())

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('Test/FINAL2.MOV')
# cam.open('https://192.168.0.3:8080/video')


#To Write 
resultVideo = cv2.VideoWriter('final_3.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30,(300, 300))



while True:
    ret, frame = cam.read()
    if ret == False:
        break
    frame_copy = frame.copy()
    frame_copy = cv2.resize(frame_copy, (300, 300))
    frame = cv2.resize(frame, (180, 180))    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.
    frame=np.expand_dims(frame, axis=0)
    pred=model.predict(frame)
    pred = np.argmax(pred, axis = 1)[0] 
    result = results[pred]
    cv2.putText(frame_copy, str(result), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('display', frame_copy)
    resultVideo.write(frame_copy)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cam.release()
cv2.destroyAllWindows()

