#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import PyPDF2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load the data
pdf_file = open(r'C:\Users\ruaak\Downloads\test 33 (1).pdf', 'rb')

pdf_reader = PyPDF2.PdfFileReader(pdf_file)
text = ''
for i in range(pdf_reader.getNumPages()):
    text += pdf_reader.getPage(i).extractText()
df = pd.read_csv(text)

# Preprocess the data
df = df[['ball_x', 'ball_y']]
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
X = df[:-1]
y = df[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(2))

# Train the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train.values.reshape((y_train.shape[0], y_train.shape[1])), epochs=50, batch_size=32)

# Evaluate the model
score = model.evaluate(X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test.values.reshape((


# In[ ]:




