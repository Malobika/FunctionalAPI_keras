# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 23:52:08 2018

@author: dipak
"""

import numpy as np
path=r'C:\Users\dipak\Desktop\newdata.csv'
with open(path) as f:
    w, h,y = [float(x) for x in next(f).split(",")]
    array= []
    
    for line in f: 
        array.append([float(x) for x in line.split(",")])

look=np.matrix(array)
print(look.shape)

array1=look[:,0]
array1=np.resize(array1,20151)

mean = array1.mean(axis=0)
array1 -= mean
std = array1.std(axis=0)
array1 /= std


from random import gauss
from random import seed
from pandas import Series
from pandas.tools.plotting import autocorrelation_plot

seed(1)
series=[gauss(0.0,1.0) for i in range(20151)]
series=Series(series)
print(series.shape)

array2=look[:,1]
array3=look[:,2]
array2=np.resize(array2,20151)
mean = array2.mean(axis=0)
array2 -= mean
std = array2.std(axis=0)
array2 /= std


array3=np.resize(array3,20151)

mean = array3.mean(axis=0)
array3 -= mean
std = array3.std(axis=0)
array3 /= std



std=np.std(array1)
mean1=np.mean(array1)
noise_signal1=np.random.normal(mean1,std,20151)
output1=array1+series


std2=np.std(array2)
mean2=np.mean(array2)
noise_signal2=np.random.normal(mean2,std,20151)
output2=array2+series

std3=np.std(array3)
mean3=np.mean(array3)
noise_signal3=np.random.normal(mean3,std,20151)
output3=array3+series


new_array=np.concatenate((array1,array2),axis=0)
final_output=np.concatenate((new_array,array3),axis=0)


new_array2=np.concatenate((output1,output2),axis=0)
final_input=np.concatenate((new_array2,output3),axis=0)
#print(np.shape(final_input),np.shape(final_output))
input_train,input_test=np.split(final_input,[60000])
print(np.shape(input_train))
print(np.shape(input_test))
y_train,y_test=np.split(final_output,[60000])
mean = input_train.mean(axis=0)
input_train -= mean
std = input_train.std(axis=0)
input_train /= std
input_test -= mean
input_test /= std

mean = y_train.mean(axis=0)
y_train -= mean
std = y_train.std(axis=0)
y_train /= std
y_test -= mean
y_test /= std

from numpy import array
input_train=input_train.reshape(1000,20,3)
input_test=input_test.reshape(453,1)
y_train=y_train.reshape(1000,20,3)
y_test=y_test.reshape(453,1)
x=np.array(input_train,dtype=np.float32)
input_test=np.array(input_test,dtype=np.float32)
y=np.array(y_train,dtype=np.float32)
y_test=np.array(y_test,dtype=np.float32)
#from keras.layers import Dense
from keras.layers import LSTM,Embedding,Conv1D
#from keras.layers import Embedding
from keras.models import Sequential
#from keras.optimizers import RMSprop
from keras.utils import to_categorical


from keras import layers
from keras import Input
from keras.models import Model
from keras.layers.merge import concatenate,add
leng=3
leng1=10
from keras.layers import Dense

posts_input = Input(shape=(20,3), dtype='float32')

Layer1Input = Conv1D(leng1, leng, activation='relu', padding='same')(posts_input)


y = Conv1D(leng1, leng, activation='relu', padding='same')(Layer1Input)
y = Conv1D(leng1, leng, activation='relu', padding='same')(y)

y = Conv1D(leng1, leng, activation='relu', padding='same')(y)
y = Conv1D(leng1, leng, activation='relu', padding='same')(y)
newlayer3=Dense(leng,activation='relu')(y)
newlayer4=Dense(leng,activation='relu')(newlayer3)

model = Model(inputs=[Layer1Input], outputs=[newlayer4])

model.compile(optimizer='adam',loss='mse',metrics=['acc'])
model.fit(x, y,epochs=10000,batch_size=64,validation_split=0.1)


model.summary()


