import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Reshape, Lambda, Dropout
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices()) 

BEFORE = 50  # num days as input
PRED = 1     # num days to predict
EPOCHS = 30
N_HIDDEN = 100

# DATA PREP
apple_file = pd.read_csv('../dataset/Stocks/aapl.us.txt')
apple_prices = []
print('apple_file ', apple_file)

for _, row in apple_file.iterrows():
    apple_prices.append(row[4])
print('len(apple_prices):', len(apple_prices))
print('apple_prices[0]:', apple_prices[0])
print('apple_prices[len(apple_prices)-1]:', apple_prices[len(apple_prices)-1])

apple_train_prices = apple_prices[0:len(apple_prices)-1000]
apple_test_prices = apple_prices[len(apple_prices)-1000:]


# NORMALIZE DATASET
mean, variance = tf.nn.moments(tf.constant(apple_train_prices, dtype=float), axes=0)
print('mean:', mean)
print('variance:', variance)
apple_train_prices = (apple_train_prices-mean)/variance
apple_test_prices = (apple_test_prices-mean)/variance
print('apple_train_prices:', apple_train_prices)
# CONSTRUCT DATA
N_TRAIN = len(apple_train_prices)-BEFORE-PRED+1
apple_train_X = []
apple_train_Y = []
for i in range(N_TRAIN):
    apple_train_X.append(apple_train_prices[i:i+BEFORE])
    x_before = apple_train_prices[i+BEFORE-1]
    y_true = apple_train_prices[i+BEFORE]
    val = [0]
    if y_true > x_before:
        val = [1]
    # to set data to predict the price
    val = [apple_train_prices[i+BEFORE]]
    apple_train_Y.append(val)
apple_train_X = np.array(apple_train_X)
apple_train_Y = np.array(apple_train_Y)
apple_train_X = np.reshape(apple_train_X, (apple_train_X.shape[0], apple_train_X.shape[1], 1))
N_TEST = len(apple_test_prices)-BEFORE-PRED+1
apple_test_X = []
apple_test_Y = []
for i in range(N_TEST):
    apple_test_X.append(apple_test_prices[i:i+BEFORE])
    x_before = apple_test_prices[i+BEFORE-1]
    y_true = apple_test_prices[i+BEFORE]
    val = [0]
    if y_true > x_before:
        val = [1]
    # to set data to predict the price
    val = [apple_test_prices[i+BEFORE]]
    apple_test_Y.append(val)
apple_test_X = np.array(apple_test_X)
apple_test_Y = np.array(apple_test_Y)
apple_test_X = np.reshape(apple_test_X, (apple_test_X.shape[0], apple_test_X.shape[1], 1))
print('apple_train_X.shape:', apple_train_X.shape)
print('apple_train_Y.shape:', apple_train_Y.shape)
print('apple_test_X.shape:', apple_test_X.shape)
print('apple_test_Y.shape:', apple_test_Y.shape)
print(apple_train_X[6999][BEFORE-3:]*variance+mean)
# print(apple_train_Y[7000]*variance+mean)
print(apple_train_Y[6999])

# BUILD MODEL
def StockMarketModel(Tx, n_a, n_values):
    x = Input(shape=(Tx, 1))

    X = LSTM(n_a, return_sequences=True)(x)
    X = Dropout(0.3)(X)
    X = LSTM(n_a)(X)
    X = Dropout(0.3)(X)
    X = Dense(20, activation='relu')(X)
    X = Dense(n_values, activation='relu')(X)
    # X = Dense(n_values, activation='sigmoid')(X)

    return Model(inputs=[x], outputs=X)

# TRAIN MODEL
model = StockMarketModel(BEFORE, N_HIDDEN, PRED)
opt = Adam()
model.compile(optimizer=opt, loss='mse')
# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit([apple_train_X], apple_train_Y, epochs=EPOCHS)

# EVAL MODEL
loss = model.evaluate(x=[apple_test_X], y=apple_test_Y)
print('test loss:', loss)

# GRAPH PREDICTIONS for direct price prediction
a0 = np.zeros((N_TEST, N_HIDDEN))
c0 = np.zeros((N_TEST, N_HIDDEN))
preds = model.predict(x=[apple_test_X])*variance+mean

plt.figure(1)
plt.plot(preds, label='predictions')
plt.plot(apple_test_Y*variance+mean, label='ground truth')
plt.title('1000 days of Apple stock prediction on test set')
plt.ylabel('stock prices')
plt.xlabel('day')
plt.legend()

plt.figure(2)
plt.plot(preds[200:250], label='predictions')
plt.plot(apple_test_Y[200:250]*variance+mean, label='ground truth')
plt.title('50 days of Apple stock prediction on test set')
plt.ylabel('stock prices')
plt.xlabel('day')
plt.legend()

plt.show()

