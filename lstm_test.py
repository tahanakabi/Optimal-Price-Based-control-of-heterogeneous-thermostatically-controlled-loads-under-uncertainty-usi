import lstm
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import numpy as np
from keras.models import load_model
import pandas as pd
import pickle
# This script is for training an LSTM network with given hyperparameters.

def train_score(X, y, i=0, model=None):

    # initialize model according to the given values of the network
    if model==None:
        model = lstm.build_model(input_shape=[seq_len, X.shape[2]],
                             lstm_size=30,
                             num_lstm=2,
                             dropout=0.2,
                             activation='tanh',
                             recurrent_activation='selu',
                             optimizer='rmsprop')
    model.fit(
        X,
        y,
        validation_split=0.2, epochs=100)
    print('Training duration (s) : ', time.time() - global_start_time)
    model.save('lstm_model'+str(i)+'.h5')

def predict(X, forecast, model, horizon, scaler):
    predictions, predicted_frames = lstm.predict_sequences_multiple(model, X, forecast, horizon)
    try:
        predicted_load = lstm.inverse_transform1(predicted_frames, scaler)
        # true_load = lstm.inverse_transform(dataset[2], dataset[3], scaler)
        # rmse = sqrt(mean_squared_error(true_load, predicted_load))
        # mape = np.mean(np.abs((true_load - predicted_load) / true_load)) * 100
    except Exception as e:
        print(e)
        # rmse=100.0
        # mape=100.0
    return predictions, predicted_load
def positive_values(x):
    return (x+np.abs(x))/2

if __name__ == '__main__':
    global_start_time = time.time()
    # # epochs  = 10
    seq_len = 2
    horizon = 6
    num_users = 30

    print('> Loading data... ')

    # X_train, y_train, X_test, y_test, scaler = lstm.load_data('fuzzy_out0.csv', seq_len)

    X_train, y_train, X_test, y_test = [], [], [], []
    _, _, _, _, scaler = lstm.load_data('fuzzy_out0.csv', seq_len)
    model = load_model('lstm_model0.h5')
    for i in range(num_users)[1:]:
        X_tr, y_tr, X_te, y_te, _ =  lstm.load_data('fuzzy_out'+str(i)+'.csv', seq_len)
        X_train.append(X_tr)
        y_train.append(y_tr)
        X_test.append(X_te)
        y_test.append(y_te)
        train_score(X_tr, y_tr,i, model=model)

    # train_score(X_train, y_train)
    # model = load_model('lstm_model0.h5')
    # dataframe = pd.read_csv('fuzzy_out0.csv')
    # data = scaler.transform(dataframe.values)
    # keep only the values in X_test
    # data = data[-X_test.shape[0]:]
    # data = data[-X_test[0].shape[0]:]

    # Single predictions
    # predictions = positive_values(lstm.predict(model, X_test))

    # rmse = sqrt(mean_squared_error(y_test, predictions))
    # mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    # print('Test RMSE: %.3f' % rmse)
    # pyplot.plot(y_test[:], label='True')
    # pyplot.plot(predictions[:], '--', label='predicted')
    # pyplot.legend()
    # pyplot.show()

    # Multiple predictions
    # multiple_predictions = []
    # for i in range(data.shape[0]-horizon, 0 ,-horizon):
    #     forecast = data[i:i+horizon, :-1]
    #     normalized_prediction, inversed_prediction = predict(X_test[i], forecast, model, horizon, scaler=scaler)
        # pyplot.plot(normalized_prediction, label='predicted')
        # pyplot.plot(y_test[i:i+horizon], label='True')
        # pyplot.legend()
        # pyplot.show()
        # multiple_predictions.insert(0,normalized_prediction)

    # print(inversed_prediction)
    # print(y_test[-48:-24])
    # Plot results
    # multiple_predictions = positive_values(np.array(multiple_predictions))
    # multiple_predictions = multiple_predictions.reshape(multiple_predictions.size)
    # y_test = y_test[y_test.size-multiple_predictions.size:]
    # pyplot.plot(y_test[:], label='True')
    # pyplot.plot(multiple_predictions[:],'--', label='predicted')
    # pyplot.legend()
    # pyplot.show()

    # with open('scaler.pickle', 'wb') as f:
    #     pickle.dump(scaler, f)