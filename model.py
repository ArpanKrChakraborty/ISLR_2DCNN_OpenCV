#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:46:24 2022

@author: adhocrobin
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow

shape = (180, 180, 3)

def threeDCNN():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=shape, kernel_regularizer=tensorflow.keras.regularizers.l1(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l1(0.01)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l1(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Conv2D(256, kernel_size=(6, 1), activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l1(0.01)))
    model.add(Conv2D(512, kernel_size=(1, 1), activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l1(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(11, activation='softmax'))
    
    return model
    
def compileModel(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model
