# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:47:27 2023

@author: igorc
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))

    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 30], 'epochs': [50, 100], 'optimizer': ['adam', 'sgd'], 
              'loss': ['binary_crossentropy', 'hinge'], 
              'kernel_initializer': ['random_uniform', 'normal'], 
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]} 

grid_search = GridSearchCV(classificador, parametros, scoring='accuracy', cv=5)
grid_search = grid_search.fit(previsores,classes)
melhores_parametros = grid_search.best_params_
melhor_precisão = grid_search.best_score_