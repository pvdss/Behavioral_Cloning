# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 21:22:41 2017

@author: Sundeep
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:30:42 2017

@author: Sundeep
"""

#importing the required packages
#csv for reading the driving log csv file
import csv
#cv2 for reading images and image operations
import cv2
#numpy for numpy array and numeric operations
import numpy as np
#shuffle to shuffle the data
from random import shuffle
#sklearn for splitting data and shuffling
import sklearn

#variable to hold the driving log
lines = []

#read the csv driving log to lines
with open("../training_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile);
    for line in reader:
        lines.append(line)

#slipt the data 
from sklearn.cross_validation import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#data generator,
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            for line in batch_samples:
                image = cv2.imread(line[0])
                measurement = float(line[3])
                images.append(image)
                measurements.append(measurement)
                image_flipped = np.fliplr(image)
                measurement_flipped = -measurement
                images.append(image_flipped)
                measurements.append(measurement_flipped)
                left_image = cv2.imread(line[1])
                left_messurement = measurement + 0.1
                images.append(left_image)
                measurements.append(left_messurement)
                right_image = cv2.imread(line[2])
                right_messurement = measurement - 0.11
                images.append(right_image)
                measurements.append(right_messurement)
            X_train = np.array(images) 
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#load the models and layers from keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda,Cropping2D, Activation,MaxPooling2D
from keras.layers.convolutional import Convolution2D


#network model
model = Sequential()

#preprocessing cropping, normalization, mean centered
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#Layer:1 convolution layer with 32 filters and 3x3 kernal, followed by relu activation function
model.add(Convolution2D(32, 3, 3,border_mode='valid'))
model.add(Activation('relu'))
#pooling and dropout to mitigate overfitting
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Layer:2 convolution layer with 20 filters and 3x3 kernal, followed by relu activation function
model.add(Convolution2D(20, 3, 3,border_mode='valid'))
model.add(Activation('relu'))
#pooling and dropout to mitigate overfitting
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Layer:3 convolution layer with 15 filters and 3x3 kernal, followed by relu activation function
model.add(Convolution2D(15, 3, 3,border_mode='valid'))
model.add(Activation('relu'))
#pooling and dropout to mitigate overfitting
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Layer:4 convolution layer with 10 filters and 3x3 kernal, followed by relu activation function
model.add(Convolution2D(10, 3, 3,border_mode='valid'))
model.add(Activation('relu'))
#pooling and dropout to mitigate overfitting
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#Flatten the network to move to fully connected layer
model.add(Flatten())

#Layer:5 64 neurons fully connected layer with relu activation function 
model.add(Dense(64))
model.add(Activation('relu'))

#Layer:6 32 neurons fully connected layer with relu activation function
model.add(Dense(32))
model.add(Activation('relu'))

#Layer:7 8 neurons fully connected layer with relu activation function
model.add(Dense(8))
model.add(Activation('relu'))

#Layer:8 1 neurons fully connected output layer
model.add(Dense(1))

#using adam optimizer
model.compile(loss='mse',optimizer='adam')
#summarizing the model
model.summary()
#fitting the model
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

#saving the model
model.save('model.h5')