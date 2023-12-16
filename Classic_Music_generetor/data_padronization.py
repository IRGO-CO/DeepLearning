import pandas as pd
import os
import pretty_midi as pm
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def extrair_dados(path):
    midi = pm.PrettyMIDI(path)
    key_signatures = [(ks.key_number, ks.time) for ks in midi.key_signature_changes]
    time_signatures = [(ts.numerator, ts.denominator, ts.time) for ts in midi.time_signature_changes]
    notas = [note.pitch for instrument in midi.instruments for note in instrument.notes]
    return key_signatures, time_signatures, notas

def normalize(data):
  mean = np.mean(data)
  var = np.var(data)
  return (data - mean) / np.sqrt(var)



diretorio_raiz = 'Midi_train'
caminho_arquivo = []
dados = []
key = []
time = []
notas = []

for compositor in os.listdir('Midi_train'):
    caminho = os.path.join(diretorio_raiz, compositor)
    for arquivo in os.listdir(caminho):
        caminho_arquivo.append(os.path.join(caminho, arquivo))
        

    
for i in caminho_arquivo:
    dados.append(extrair_dados(i))
    print('Extract from : ', i)

for i in range(0, 295):
    key.append(dados[i][0])
    time.append(dados[i][1])
    notas.append(dados[i][2])

data_key = {}
max_columns = 11

for i, sublist in enumerate(key):
    for j in range(max_columns):
        print(f'j  = {j}')       
        column_name = f'key_tuple_{j + 1}'
    
        if column_name not in data_key:
            data_key[column_name] = []      
        if j < len(sublist):
            data_key[column_name].append(sublist[j])
        else:
            data_key[column_name].append((0.0, 0.0))
            



data_time = {}

for i, sublist in enumerate(time):
    for j in range(max_columns):
        print(f'j  = {j}')       
        column_name = f'time_tuple_{j + 1}'
    
        if column_name not in data_time:
            data_time[column_name] = []      
        if j < len(sublist):
            data_time[column_name].append(sublist[j])
        else:
            data_time[column_name].append((0.0, 0.0))


dicionario_completo = {**data_key, **data_time,'notes': notas[:]}


df = pd.DataFrame(dicionario_completo)
df['notes'] = df['notes'].apply(tuple)



df.to_csv('data_set.csv', index=False)


