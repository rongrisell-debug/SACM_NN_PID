#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
NN_for_PID vers 1
to classify time series used in Driver_NN_PID, based on
https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
See SimpleNN for correct original with
10 inputs, 3 outputs
"""
import numpy as np
from numpy import array
import csv
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

data = []
filename = ""
try:
    with open("training.csv", 'r', newline = '', encoding = "utf8") as inp:
        csv_in = csv.reader(inp, delimiter=',')
        for row in csv_in:
            #print(row)
            data.append(row)
    inp.close
except csv.Error as e:
    sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
    print("***Input file couldn't not be openned")
        
finally:
    inp.close
    print("test completed normally.")

dataset = array(data)
xy = dataset.shape
print("data.shape = ",xy)

# separate data
split = 50
train, test = dataset[:split,:], dataset[split:,:]
print("MAIN CODE: train----------------")
print(train)
print("MAIN CODE: test----------------")
print(test)
print("==========================================")
print(" ")
print(" As it is split:")
train_x = train[:,0:8]
##train_x = train[:,0:6]
train_y= train[:,-4:-1]

print(" ")
print("train_x ----------------")
print(train_x)
print("train_y ----------------")
print(train_y)

print(" ")
print(" test_x,----------------")
test_x = test[:,0:8]
#####test_x = test[:,0:6]
print(test_x)
print(" test_y,----------------")
test_y = test[:,-4:-1]
print(test_y)
print(" ")
# THIS IS TRICKY - CREATED ERROR IN SIMILAR CODE ??
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model
model = Sequential()

#####model.add(Dense(10, input_shape=(6,), activation='relu', name='fc1'))
model.add(Dense(10, input_shape=(8,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())
'''Layer (type)                 Output Shape              Param #   
=================================================================
fc1 (Dense)                  (None, 10)                50        
_________________________________________________________________
fc2 (Dense)                  (None, 10)                110       
_________________________________________________________________
output (Dense)               (None, 3)                 33        
=================================================================
Total params: 193
Trainable params: 193
Non-trainable params: 0'''

# Train the model
model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

# Test on unseen data
results = model.evaluate(test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

