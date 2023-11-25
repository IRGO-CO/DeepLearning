# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:29:30 2023

@author: Igor
"""

import numpy as np
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma)) 

def derivadaSig(sig):
    return sig * (1 - sig)

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

saidas = np.array([[0],[1],[1],[0]])
pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])
pesos1 = np.array([[-0.017], [-0.893], [0.148]])
epocas = 100
taxaAprendizagem = 0.3
momento = 1 

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erros = saidas - camadaSaida
    mediaErros = np.mean(np.abs(erros))
    
    derivadaSaida = derivadaSig(camadaSaida)
    deltaSaida = erros * derivadaSaida
    
    deltaSaidaXPeso = deltaSaida.dot(pesos1.T)
    deltaCamadaOculta = deltaSaidaXPeso * derivadaSig(camadaOculta)
    
    novoPeso = camadaOculta.T.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (novoPeso * taxaAprendizagem)
    