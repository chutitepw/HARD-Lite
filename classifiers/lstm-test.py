import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

tf.compat.v1.random.set_random_seed(1)

import numpy as np
from utils import findfile, load_file_anomaly, create_sequences,load_scaler, threshold_calculation, threshold_load, plot_anomaly

TIME_STEPS=50

scaler = load_scaler('lstm-scaler.sav')

#### Find Files for Training/Testing ####
test_path = findfile('benign-*.csv', '../data/test')
test_path2 = findfile('ransom-*.csv', '../data/test')

X_test, y_test, scaler = load_file_anomaly(test_path, TIME_STEPS, 0, scaler)
X_test2, y_test2, scaler = load_file_anomaly(test_path2, TIME_STEPS, 0, scaler)

print("Testing Benign Data Shape: ", X_test.shape,y_test.shape)
print("Testing Malicious Data Shape: ", X_test2.shape,y_test2.shape)

model = tf.keras.models.load_model('lstm-model.keras')

threshold = threshold_load('threshold.txt')

#### Test Model ####
X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.abs(X_test_pred-y_test)

plot_anomaly(test_mae_loss, X_test_pred.shape[1], threshold)

X_test_pred2 = model.predict(X_test2, verbose=0)
test_mae_loss2 = np.abs(X_test_pred2-y_test2)

plot_anomaly(test_mae_loss2, X_test_pred2.shape[1], threshold)
