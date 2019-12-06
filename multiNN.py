#import de bibliotecas

import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


#importando os dados

## 3JHN
pickle_in = open("X1_norm_STDscaler.pickle","rb")
X1  = pickle.load(pickle_in)

pickle_in = open("y1_n_norm.pickle","rb")
y1  = pickle.load(pickle_in)

##3JHH
pickle_in = open("X2_norm_STDscaler.pickle","rb")
X2  = pickle.load(pickle_in)

pickle_in = open("y2_n_norm.pickle","rb")
y2  = pickle.load(pickle_in)

#diferentes tipos numero de layers e tamanhos para cada layer
dense_layers = [3,4,5]
layer_sizes  = [32,64,128]


#cria varias redes neurais para  a matriz X de features e y de labels
def create_multiple_Nets(X,y,error_function,tipo):

	for dense_layer in dense_layers:
		for layer_size in layer_sizes:
			NAME = "tipo-{}-{}-nodes-{}-layers".format(tipo, layer_size, dense_layer, int(time.time()))
			print(NAME)

			model = Sequential()
			model.add(tf.keras.layers.Dense(layer_size, activation=tf.nn.relu, input_shape = X.shape[1:]))#input layer
			for _ in range(dense_layer):
				model.add(tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)) #hidden layers
			
			model.add(tf.keras.layers.Dense(1, activation='linear')) #output layer
			tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
			model.compile(loss=error_function,
                          optimizer='adam',
                          metrics=['mae'],
                          )
			model.fit(X, y,
                      batch_size=10,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])
			model.save("modelo-{}".format(NAME))
				




################################# criando todas as redes ###################################

## 3JHN ##
create_multiple_Nets(X1,y1,'mean_squared_error',"3JHN")

## 3JHH ##

create_multiple_Nets(X2,y2,'mean_absolute_error',"3JHH")