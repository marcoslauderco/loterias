# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
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
epochs_interations = 256
batch_size = 256

data_path = "./megasena_data"
table_MN = pd.read_html('megasena.html')
df = table_MN[0]
del df["Cidade"]
del df["Rateio Faixa 1"]
del df["Rateio Faixa2"]
del df["Rateio Faixa 3"]
del df["Valor Arrecadado"]
del df["Estimativa para o prÃ³ximo concurso"]
del df["Valor Acumulado PrÃ³ximo Concurso"]
del df["Acumulado"]
del df["Sorteio Especial"]
del df["ObservaÃ§Ã£o"]
del df["Data do Sorteio"]
del df["Ganhadores Faixa 1"]
del df["Ganhadores Faixa 2"]
del df["Ganhadores Faixa 3"]
del df["Concurso"]

df = df[df.Coluna1.notnull()]
df = df[df.Coluna2.notnull()]
df = df[df.Coluna3.notnull()]
df = df[df.Coluna4.notnull()]
df = df[df.Coluna5.notnull()]
df = df[df.Coluna6.notnull()]

# ultimos resultado
df = df.tail(256)

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
model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs_interations, verbose=1)

# salvando o modelo
model.save(data_path)

to_predict = df.tail(window_length)
to_predict = np.array(to_predict)

scaled_to_predict = scaler.transform(to_predict)

y_pred = model.predict(np.array([scaled_to_predict]))

print("Previsto:", scaler.inverse_transform(y_pred).astype(int)[0])