import numpy as np
import matplotlib.pyplot as plt
import os
import keras
from stock_models import StockMarketModels
from tensorflow.python.client import device_lib
from keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(device_lib.list_local_devices())


BEFORE = 50     # num days as input
PRED = 1        # num days to predict
EPOCHS = 10     # num epochs to train
N_HIDDEN = 100  # num hidden states in LSTM
DATA_DIR = '../dataset_prepared/'   # the directory of the dataset

# Training a model on the Apple stock prices from 1987-2017
def main():
    apple_train_X, apple_train_Y, apple_test_X, apple_test_Y = load_data('logistic')
    mean_variance = load_nparray(DATA_DIR + 'mean_variance')
    mean = mean_variance[0]
    variance = mean_variance[1]

    '''
    loss = train_binary_model(apple_train_X, apple_train_Y, apple_test_X, apple_test_Y)
    print('test loss:', loss)
    '''

    loss, preds = train_logistic_model(apple_train_X, apple_train_Y, apple_test_X, apple_test_Y, mean, variance)
    print('test loss:', loss)
    plot_logistic_model(preds, apple_test_Y, mean, variance)


def load_nparray(filename):
    with open(filename, 'rb') as f:
        return np.load(f)


# type_data should be 'binary' or 'logistic'
def load_data(type_data):
    apple_train_X = load_nparray(DATA_DIR + type_data + '_apple_train_X')
    apple_train_Y = load_nparray(DATA_DIR + type_data + '_apple_train_Y')
    apple_test_X = load_nparray(DATA_DIR + type_data + '_apple_test_X')
    apple_test_Y = load_nparray(DATA_DIR + type_data + '_apple_test_Y')
    return apple_train_X, apple_train_Y, apple_test_X, apple_test_Y


# this model is to predict the exact next day stock prices.
def train_logistic_model(apple_train_X, apple_train_Y, apple_test_X, apple_test_Y, mean, variance):
    # change this line to change different logistic models
    model = StockMarketModels().LSTMLogisticModel(BEFORE, N_HIDDEN, PRED)
    opt = Adam()
    model.compile(optimizer=opt, loss='mse')
    model.fit(apple_train_X, apple_train_Y, epochs=EPOCHS)
    model.summary()

    loss = model.evaluate(x=apple_test_X, y=apple_test_Y)
    preds = model.predict(x=apple_test_X)*variance+mean

    return loss, preds


# this model is to predict whether the next day stock price will go up or down
def train_binary_model(apple_train_X, apple_train_Y, apple_test_X, apple_test_Y):
    model = StockMarketModels().LSTMBinaryModel(BEFORE, N_HIDDEN, PRED)
    opt = Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(apple_train_X, apple_train_Y, epochs=EPOCHS)

    loss = model.evaluate(x=apple_test_X, y=apple_test_Y)

    return loss


# plot the stock price predictions
def plot_logistic_model(preds, apple_test_Y, mean, variance):
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

main()
