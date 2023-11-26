import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score



previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes, test_size= 0.25)

otimizador = keras.optimizers.Adam(learning_rate = 0.0001, clipvalue = 0.5, weight_decay=0.0001)

classificador = Sequential()
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim = 30))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

tensorboard_callback = TensorBoard(log_dir='./logs')
classificador.fit(previsores_treinamento, classes_treinamento, batch_size=10, epochs=100, callbacks=[tensorboard_callback])

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classes_teste, previsoes)
matriz = confusion_matrix(classes_teste,previsoes)
                          