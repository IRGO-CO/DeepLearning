# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:47:27 2023

@author: igorc

Dados fornecidos por: UC Irvine Machine Learning Repository
Autor: UCI
Disponível em: https://archive.ics.uci.edu/dataset/14/breast+cancer
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

parametros = {'batch_size': [10, 30], 
              'epochs': [50, 100] ,
               'optimizer': ['adam', 'sgd'],
               'loos': ['binary_crossentropy', 'hinge'],
               'kernel_initializer': ['random_uniform', 'normal'],
               'activation': ['relu', 'tanh'],
               'neurons': [16, 8]
              }

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))

    classificador.compile(optimizer=optimizer, loss=loos, metrics=['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criarRede)


grid_search = GridSearchCV(estimator=classificador, 
                           param_grid=parametros, 
                           scoring='accuracy', 
                           cv=5)
grid_search = grid_search.fit(previsores, classes)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
