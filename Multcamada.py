# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:29:30 2023

@author: Igor
"""

import numpy as np


from sklearn import datasets



def sigmoid(soma):
    return 1 / (1 + np.exp(-soma)) 

def derivadaSig(sig):
    return sig * (1 - sig)



base = datasets.load_breast_cancer()
pesos0 = 2*np.random.random((30, 3)) - 1
pesos1 = 2*np.random.random((3, 1)) - 1
epocas = 1000000
taxaAprendizagem = 0.3
momento = 1 
entradas = base.data
s = base.target
saidas = np.empty([569, 1], dtype = int)
for i in range(569):
    saidas[i] = s[i]

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erros = saidas - camadaSaida
    mediaErros = np.mean(np.abs(erros))
    print('Erro: ' + str(mediaErros))
    
    derivadaSaida = derivadaSig(camadaSaida)
    deltaSaida = erros * derivadaSaida
    
    pesos1T = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1T)
    deltaCamadaOculta = deltaSaidaXPeso * derivadaSig(camadaOculta)
    
    camadaOcultaT = camadaOculta.T
    novoPeso1 = camadaOcultaT.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (novoPeso1 * taxaAprendizagem)
    
    camadaEntradaT = camadaEntrada.T
    novoPeso0 = camadaEntradaT.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (novoPeso0 * taxaAprendizagem)