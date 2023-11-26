# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 07:15:10 2023

@author: igorc
"""
# import pybrain
from pybrain.structure import feedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = feedForwardNetwork()

entryLayer = LinearLayer(2)
ocultLayer = SigmoidLayer(3)
exitLayer = SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(entryLayer)
rede.addModule(ocultLayer)
rede.addModule(exitLayer)
rede.addModule(bias1)
rede.addModule(bias2)

entryToOcult = FullConnection(entryLayer, ocultLayer)
ocultToExit = FullConnection(ocultLayer, exitLayer)
biasToOcult = FullConnection(bias1, ocultLayer)
biasToExit = FullConnection(bias2, exitLayer)

rede.sortModules()

print(rede)
print(ocultLayer.params)
print(ocultToExit.params)
print(biasToOcult.params)
print(biasToExit.params)