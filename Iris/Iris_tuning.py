# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:19:20 2023

@author: Igor
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.data.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
lableencoder = LabelEncoder()

classe = lableencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

parametros = {'batch_size': [10, 50, 100], 
              'epochs': [500, 1000, 5000] ,
               'optimizer': ['adam', 'sgd', 'relu'],
               'loos': ['categorical_crossentropy', 'hinge', 'categorical_hinge'],
               'kernel_initializer': ['random_uniform', 'normal', 'constant'],
               'activation': ['relu', 'tanh', 'elu'],
               'neurons': [4, 8, 16]
              }

def criar_rede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units=neurons, activation=activation, input_dim=4 ))
    classificador.add(Dense(units=neurons, activation=activation))
    classificador.add(Dense(units=3, activation='softmax'))
    classificador.compile(optimizer=optimizer, loss=loos, metrics=['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criar_rede)

grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=10)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_