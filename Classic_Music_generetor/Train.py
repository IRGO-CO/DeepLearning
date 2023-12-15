# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:54:54 2023

@author: igorc
"""


import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


base = pd.read_csv('data_set.csv')


    