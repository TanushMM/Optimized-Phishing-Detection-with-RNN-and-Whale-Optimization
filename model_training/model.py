from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    print("\n Model Summary")
    model.summary()
    return model
