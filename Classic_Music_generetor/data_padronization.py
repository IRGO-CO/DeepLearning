import pandas as pd
import os
import pretty_midi as pm
import numpy as np

def extrair_dados(path):
    midi = pm.PrettyMIDI(path)
    key_signatures = [(ks.key_number, ks.time) for ks in midi.key_signature_changes]
    time_signatures = [(ts.numerator, ts.denominator, ts.time) for ts in midi.time_signature_changes]
    notas = [note.pitch for instrument in midi.instruments for note in instrument.notes]
    return key_signatures, time_signatures, notas


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
    print('Extraido de : ', i)

for i in range(0, 295):
    key.append(dados[i][0])
    time.append(dados[i][1])
    notas.append(dados[i][2])
    
Frame = pd.DataFrame(
    {
        "Key_signature": key,
        "Time_signature": time,
        "Notas": notas,
        
    }
)

print(key)

Frame.to_csv('data_set.csv', index=False)

Frame.head()

base = pd.read_csv('data_set.csv')

