# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 02:46:34 2023

@author: igorc
"""


import numpy as np
import struct
from music21 import converter, pitch



def bits64_f(numero_float):
    representacao_binaria = format(struct.unpack(
        '!Q', struct.pack('!d', numero_float))[0], '064b')
    return representacao_binaria


def bits64(lista_numeros):
    binarios = [format(x, '08b') for x in lista_numeros]
    binario_completo = ''.join(binarios)
    binario_completo = binario_completo.rjust(64, '0')
    return binario_completo

def reverter_bits64(binario_completo):
    blocos = [binario_completo[i:i+8] for i in range(0, len(binario_completo), 8)]
    blocos_validos = [bloco for bloco in blocos if '1' in bloco]
    numeros = [int(bloco, 2) for bloco in blocos_validos]

    return numeros


midi_file_path = "D:\Repositorys\DeepLearning\Estudos\Classic_Music_generetor\music21_tests\Dataset\midis\OXC7Fd0ZN8o.mid"
midi_stream = converter.parse(midi_file_path)


informacoes_numericas = []

for elemento in midi_stream.flat:
    if 'Note' in elemento.classes:
        altura = pitch.Pitch(elemento.pitch).midi
        duracao = elemento.duration.quarterLength
        inicio = elemento.offset
        velocidade = elemento.volume.velocity if 'Volume' in elemento.volume.classes else None
        informacoes_numericas.append(
            {'Tipo': 'Nota', 'Altura': altura, 'Duração': duracao, 'Início': inicio, 'Velocidade': velocidade})
    elif 'Chord' in elemento.classes:
        alturaa = [pitch.Pitch(n.pitch).midi for n in elemento.notes]
        duracao_acorde = elemento.duration.quarterLength
        inicio_acorde = elemento.offset
        velocidade_acorde = elemento.volume.velocity if 'Volume' in elemento.volume.classes else None
        informacoes_numericas.append({'Tipo': 'Acorde', 'Alturas': alturaa, 'Duração': duracao_acorde,
                                     'Início': inicio_acorde, 'Velocidade': velocidade_acorde})

for info in informacoes_numericas:
    print('\nExito: ', info)

features = []
for info in informacoes_numericas:
    if info['Tipo'] == 'Nota':
        altura = bits64([info['Altura']])
        duracao = bits64_f(info['Duração'])
        inicio = bits64_f(info['Início'])
        velocidade = bits64([info['Velocidade']])
        features.append([alturaa, duracao, inicio, velocidade])
    elif info['Tipo'] == 'Acorde':
        alturaa = bits64(info['Alturas'])
        duracao = bits64_f(info['Duração'])
        inicio_acorde = bits64_f(info['Início'])
        velocidade_acorde = bits64([info['Velocidade']])
        features.append([alturaa, duracao, inicio_acorde, velocidade_acorde])


features_array = np.array(features)

teste=bits64([50,30,10,5,80])
print(f'Conversão para bits: {teste}')
revertest=reverter_bits64(teste)
print(f'Valor original{revertest}')
