import keras
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Reshape, Lambda, Dropout, Flatten

# Models used for stock market price prediction
class StockMarketModels():
    # Simple dense FC model to start out
    def SimpleVanillaModel(self, Tx, n_values):
        return keras.Sequential([
            Flatten(),
            Dense(512, activation='relu'),
            Dense(2048, activation='relu'),
            Dense(n_values, activation='relu')
        ])


    # Try a simple LSTM model
    def SimpleLSTMModel(self, Tx, n_a, n_values):
        return keras.Sequential([
            LSTM(n_a),
            Dense(n_values, activation='relu')
        ])


    # Built up a good LSTM model with dropout
    def LSTMLogisticModel(self, Tx, n_a, n_values):
        return keras.Sequential([
            LSTM(n_a, return_sequences=True),
            Dropout(0.3),
            LSTM(n_a),
            Dropout(0.3),
            Dense(20, activation='relu'),
            Dense(n_values, activation='relu')
        ])


    # Same as above logistic model except with final sigmoid activation
    def LSTMBinaryModel(self, Tx, n_a, n_values):
        return keras.Sequential([
            LSTM(n_a, return_sequences=True),
            Dropout(0.3),
            LSTM(n_a),
            Dropout(0.3),
            Dense(20, activation='relu'),
            Dense(n_values, activation='sigmoid')
        ])
