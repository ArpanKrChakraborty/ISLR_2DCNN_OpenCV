#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:51:59 2022

@author: adhocrobin
"""

import model as my_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import warnings

from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')

# Constants
SIZE = (180, 180)
CHANNELS = 3
BS = 35
EPOCHS=5
DIRECTORY='DATA/'

location='DATA/*'
classes = [i.split(os.path.sep)[1] for i in glob.glob(location)]
classes.sort()

glob_pattern='data/{classname}/*.jpg'

data_aug = ImageDataGenerator(
            zoom_range=.1,
            rotation_range=8,
            width_shift_range=.2,
            height_shift_range=.2,
            rescale=1./255,
            horizontal_flip=False,
            fill_mode='nearest',
            validation_split=.3)

train = data_aug.flow_from_directory(batch_size=BS,
                                     directory=DIRECTORY,
                                     shuffle=True,
                                     target_size=SIZE,
                                     subset='training',
                                     class_mode='categorical')

validation = data_aug.flow_from_directory(batch_size=BS,
                                     directory=DIRECTORY,
                                     shuffle=True,
                                     target_size=SIZE,
                                     subset='validation',
                                     class_mode='categorical')

# model = model_one.compileModel(model_one.threeDCNN())
model = my_model.compileModel(my_model.threeDCNN())

history=model.fit(
    train,
    validation_data=validation,
    epochs=EPOCHS,
    steps_per_epoch=150,
    validation_steps=20
)

model.save('luck-two')

np.save('luck-two.npy',history.history)

# history=np.load('model-one-history-one.npy',allow_pickle='TRUE').item()
# 6 is the best
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

