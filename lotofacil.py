# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow import keras

def isEmptyDir(path):
    if os.path.exists(path) and not os.path.isfile(path):
  
        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        print("The path is either for a file or not valid")
        return False

window_length = 7
epochs_interations = 5
batch_size = 25

data_path = "./lotofacil_data"
table_MN = pd.read_html('lotofacil.html')

df = table_MN[0]
del df["Concurso"]
del df["Data_Sorteio"]
del df["ArrecadaÃ§Ã£o_Total"]
del df["Ganhadores_15_NÃºmeros"]
del df["Cidade"]
del df["Ganhadores_14_NÃºmeros"]
del df["Ganhadores_13_NÃºmeros"]
del df["Ganhadores_12_NÃºmeros"]
del df["Ganhadores_11_NÃºmeros"]
del df["Valor_Rateio_15_NÃºmeros"]
del df["Valor_Rateio_14_NÃºmeros"]
del df["Valor_Rateio_13_NÃºmeros"]
del df["Valor_Rateio_12_NÃºmeros"]
del df["Valor_Rateio_11_NÃºmeros"]
del df["Acumulado_15_NÃºmeros"]
del df["Estimativa_PrÃªmio"]
del df["Valor_Acumulado_Especial"]

df = df[df.Bola1.notnull()]
df = df[df.Bola2.notnull()]
df = df[df.Bola3.notnull()]
df = df[df.Bola4.notnull()]
df = df[df.Bola5.notnull()]
df = df[df.Bola6.notnull()]
df = df[df.Bola7.notnull()]
df = df[df.Bola8.notnull()]
df = df[df.Bola9.notnull()]
df = df[df.Bola10.notnull()]
df = df[df.Bola11.notnull()]
df = df[df.Bola12.notnull()]
df = df[df.Bola13.notnull()]
df = df[df.Bola14.notnull()]
df = df[df.Bola15.notnull()]

scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

number_of_rows = df.values.shape[0]
number_of_features = df.values.shape[1]

X = np.empty([ number_of_rows - window_length, window_length, 
number_of_features], dtype=float)
y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)
for i in range(0, number_of_rows-window_length):
    X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
    y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

if(isEmptyDir(data_path)):
    model = Sequential()
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = True)))
    model.add(Bidirectional(LSTM(240, input_shape = (window_length, number_of_features), return_sequences = False)))
    model.add(Dense(59))
    model.add(Dense(number_of_features))
else:
    model = keras.models.load_model(data_path)

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])
model.fit(x=X, y=y, batch_size=100, epochs=epochs_interations, verbose=1)

model.save(data_path)

to_predict = df.tail(window_length)
to_predict = np.array(to_predict)

scaled_to_predict = scaler.transform(to_predict)

y_pred = model.predict(np.array([scaled_to_predict]))

print("Previsto:", scaler.inverse_transform(y_pred).astype(int)[0])