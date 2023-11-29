# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:53:37 2023

@author: igorc

Dados fornecidos por: UC Irvine Machine Learning Repository - Iris
Autor: UCI
Disponível em: https://archive.ics.uci.edu/dataset/53/iris
"""
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix


base = pd.read_csv('iris.data.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
    previsores, classe_dummy, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units=4, activation='relu', input_dim=4))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax'))
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
                      'categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size=10, epochs=1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)




