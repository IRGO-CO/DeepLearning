# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:02:12 2023

@author: igorc

Dados fornecidos por: UC Irvine Machine Learning Repository
Autor: UCI
Disponível em: https://archive.ics.uci.edu/dataset/14/breast+cancer

"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

classificador = Sequential()
classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', 
                    kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu', 
                    kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                  metrics = ['binary_accuracy'])

classificador.fit(previsores, classes, batch_size=10, epochs=100)

entrada = np.array([[20.5, 13.85, 15.24, 90.37, 578.1, 0.1054, 0.07658, 0.05572, 0.03946, 0.1982, 0.06237, 0.2785, 0.8152, 2100.0, 24.78, 0.01037, 0.0172, 0.02794, 0.01536, 0.02279, 0.002768, 16.05, 18.92, 105.3, 746.8, 152.4, 0.1897, 245.2, 0.1367, 0.3085]])
resultado = classificador.predict(entrada)
resultado = (resultado > 0.5)
print(classificador)