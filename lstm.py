import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
## This is a library of methods to build a reccurent neural net with long-short term memory layers.
## These methods are built to handle data of pairs (x,y) and used mainly to predict the next 60 positions of a Robot given a positions

def load_data(filename, seq_len, validation_percentage=0.1):
    dataframe = pd.read_csv(filename)
    data=dataframe.values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data)
    data = scaler.transform(data)

    n_features = data.shape[1]
    X_y = series_to_supervised(data,seq_len,1)
    print(X_y.head())

    X_y = X_y.values

    row = int(round((1-validation_percentage) * X_y.shape[0]))
    n_inputs = seq_len*n_features
    train = X_y[:row, :]
    test = X_y[row:, ]

    X_train, y_train = train[:,0:n_inputs], train[:,-1]
    X_test, y_test = test[:, 0:n_inputs], test[:, -1]

    X_train = X_train.reshape((X_train.shape[0], seq_len, n_features))
    X_test = X_test.reshape((X_test.shape[0], seq_len, n_features))

    return [X_train, y_train, X_test, y_test, scaler ]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def get_last_sequence(filename, seq_len, normalise_window):
    ## read a txt file and return the last sequence of (x,y) positions
    data = []
    with open(filename, 'r') as d:  # using the with keyword ensures files are closed properly
        for line in d.readlines():
            parts = line.split(',')
            data.append([int(parts[0]), int(parts[1].replace('\n', ''))])

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    dresult = np.array(result)
    if normalise_window:
        result = normalise_windows(result)
    result = np.array(result)

    return result[-1],dresult[-1]


# def normalise_windows(window_data):
#     ## normalising data to get convergence in training
#     normalized_data = []
#     for window in window_data:
#         normalised_window = [[((float(p[0]) / float(window[0][0])) - 1),((float(p[1]) / float(window[0][1])) - 1)] for p in window]
#         normalized_data.append(normalised_window)
#     return normalized_data

def build_model(input_shape,lstm_size,num_lstm=1,optimizer='rmsprop', dropout=0.0, activation='tahnh', recurrent_activation='hard_sigmoid'):
    ## We will build a recurrent neural network with long short term memory considering the positions as a time series
    ## We use Keras to build the KNN and LSTM layers
    model = Sequential()
    for i in range(num_lstm-1):
        model.add(LSTM(lstm_size, input_shape=(input_shape[0],input_shape[1]),return_sequences=True,activation=activation,recurrent_activation=recurrent_activation))
    model.add(LSTM(lstm_size, input_shape=(input_shape[0], input_shape[1])))

    # model.add(Dropout(0.2))
    # model.add(LSTM(
    #     layers[2],
    #     return_sequences=False))

    # model.add(Dropout(0.2))
    model.add(Dense(output_dim=1))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer=optimizer)
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_sequences_multiple(model, data, forecast, prediction_len):
    ## Predict sequence of prediction_len steps before shifting prediction run forward by prediction_len steps
    ## d_xx stand for denormalized variables we needed to denormalise everything in order to predict the real positions
    # forecasted_frames = series_to_supervised(forecast, n_in=0, n_out=0)
    forecast = np.reshape(forecast, [forecast.shape[0], 1, 2])
    # prediction_seqs = []
    predicted_frames = []
    curr_frame = data
    predicted = []
    for j in range(prediction_len):
        prediction = model.predict(curr_frame[newaxis, :, :])
        predicted.append(prediction)
        curr_frame = curr_frame[1:]
        next_frame = np.append(forecast[j],predicted[-1], axis=1)
        curr_frame = np.append(curr_frame, next_frame, axis=0)
        predicted_frames.append(curr_frame)

    return np.reshape(predicted,[np.array(predicted).shape[0],1]), np.array(predicted_frames)

def predict(model, data):
    # make a prediction
    yhat = model.predict(data)

    return yhat

def inverse_transform(data,y, scaler):
    feautures = data.shape[1] * data.shape[2]
    data = data.reshape((data.shape[0], feautures))
    y = y.reshape(y.shape[0],1)
    # invert scaling for forecast
    inv_y = np.concatenate((data[:, -3:-1],y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]
    return inv_y

def inverse_transform1(data, scaler):
    feautures = data.shape[1] * data.shape[2]
    data = data.reshape((data.shape[0], feautures))
    # y = y.reshape(y.shape[0],1)
    # invert scaling for forecast
    # inv_y = np.concatenate((data[:, -3:],y), axis=1)
    inv_y = scaler.inverse_transform(data[:, -3:])
    inv_y = inv_y[:, -1]
    return inv_y