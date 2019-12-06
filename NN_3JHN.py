"""
Descrição: Gera dois modelos diferentes de redes neurais
incorporando Dropout e Leaky ReLU

Os modelos desenvolvidos são previstos para prever o acoplamento
do tipo 3JHN, para

"""
#import de bibliotecas

import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D,Flatten

#importando os dados
pickle_in = open("X1_norm_STDscaler.pickle","rb")
X1  = pickle.load(pickle_in)
pickle_in = open("y1_n_norm.pickle","rb")
y1  = pickle.load(pickle_in)

##################### MODELOS ###############################################

for j in range(0,2):
	Name = "modelo-{}".format(str(j)) #um nome único para cada rede neural

	model = Sequential()
	model.add(tf.keras.layers.Dense(256,  input_shape = X1.shape[1:]))
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dropout(0.4))

	model.add(tf.keras.layers.Dense(1024))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))
	model.add(tf.keras.layers.Dropout(0.2))

	model.add(tf.keras.layers.Dense(1024))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))
	model.add(tf.keras.layers.Dropout(0.2))

	model.add(tf.keras.layers.Dense(512))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))
	model.add(tf.keras.layers.Dropout(0.4))

	model.add(tf.keras.layers.Dense(512))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))

	model.add(tf.keras.layers.Dense(256))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))
	model.add(tf.keras.layers.Dropout(0.4))

	if j ==1:
		model.add(tf.keras.layers.Dense(128))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))
		model.add(tf.keras.layers.Dropout(0.2))

		model.add(tf.keras.layers.Dense(128))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))

		model.add(tf.keras.layers.Dense(64))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU(alpha = 0.05))

	#output layer
	model.add(tf.keras.layers.Dense(1, activation='linear'))

	tensorboard = TensorBoard(log_dir="logs/{}".format(Name)) #define o diretorio onde serão salvas as informações do processo de treinamento
	
	model.compile(loss='mean_squared_error',
		                          optimizer='adam',
		                          metrics=['mae'],		
		                          )
	model.fit(X1, y1, batch_size=10, epochs=3, validation_split=0.3, callbacks=[tensorboard])
	model.save("modelo-{}".format((str(j))))


################################################################################################################################
