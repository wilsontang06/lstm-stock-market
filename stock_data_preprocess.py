import numpy as np
import pandas as pd
import tensorflow as tf

# Pre-process the stock market price data
# This project is only using the Apple stock prices

BEFORE = 50 # num days as input
PRED = 1    # num days to predict

# DATA PREP
apple_file = pd.read_csv('../dataset/Stocks/aapl.us.txt')
apple_prices = []

for _, row in apple_file.iterrows():
    apple_prices.append(row[4])
apple_train_prices = apple_prices[0:len(apple_prices)-1000]
apple_test_prices = apple_prices[len(apple_prices)-1000:]


# NORMALIZE DATASET
mean, variance = tf.nn.moments(tf.constant(apple_train_prices, dtype=float), axes=0)
apple_train_prices = (apple_train_prices-mean)/variance
apple_test_prices = (apple_test_prices-mean)/variance
# CONSTRUCT DATA
def construct_data(N, apple_prices):
    apple_X = []
    apple_Y = []
    for i in range(N):
        apple_X.append(apple_prices[i:i+BEFORE])
        x_before = apple_prices[i+BEFORE-1]
        y_true = apple_prices[i+BEFORE]
        # to set the data to predict up/down
        value = [0]
        if y_true > x_before:
            value = [1]
        # to set data to predict the price
        # ONLY have this line uncommented when creating dataset for logistic loss
        # value = [apple_prices[i+BEFORE]]
        apple_Y.append(value)
    apple_X = np.array(apple_X)
    apple_Y = np.array(apple_Y)
    apple_X = np.reshape(apple_X, (apple_X.shape[0], apple_X.shape[1], 1))
    return apple_X, apple_Y

N_TRAIN = len(apple_train_prices)-BEFORE-PRED+1
N_TEST = len(apple_test_prices)-BEFORE-PRED+1
apple_train_X, apple_train_Y = construct_data(N_TRAIN, apple_train_prices)
apple_test_X, apple_test_Y = construct_data(N_TEST, apple_test_prices)


# VERIFY DATASETS
print('apple_train_X.shape:', apple_train_X.shape)
print('apple_train_Y.shape:', apple_train_Y.shape)
print('apple_test_X.shape:', apple_test_X.shape)
print('apple_test_Y.shape:', apple_test_Y.shape)
print(apple_train_X[6999][BEFORE-3:]*variance+mean)
# print(apple_train_Y[7000]*variance+mean)
print(apple_train_Y[6999])


# SAVE DATASET TO FILES
def save_to_file(filename, nparray):
    with open(filename, 'wb') as f:
        np.save(f, nparray)

DATA_DIR = '../dataset_prepared/'
save_to_file(DATA_DIR + 'binary_apple_train_X', apple_train_X)
save_to_file(DATA_DIR + 'binary_apple_train_Y', apple_train_Y)
save_to_file(DATA_DIR + 'binary_apple_test_X', apple_test_X)
save_to_file(DATA_DIR + 'binary_apple_test_Y', apple_test_Y)
save_to_file(DATA_DIR + 'mean_variance', np.array([mean, variance]))
