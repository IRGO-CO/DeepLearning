# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:53:30 2023

@author: igorc
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

def criarRede():
    otimizador = keras.optimizers.Adam(learning_rate = 0.0001, clipvalue = 0.5, decay=0.0001)

    classificador = Sequential()
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim = 30))
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dense(units=1, activation='sigmoid'))

    classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)

resultados = cross_val_score(classificador, X=previsores, y=classes, cv=10, scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()