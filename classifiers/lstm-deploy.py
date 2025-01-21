import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

tf.compat.v1.random.set_random_seed(1)

import os
import time
import numpy as np
from utils import findfile, load_file_deploy, create_sequences,load_scaler, threshold_calculation, threshold_load, plot_deployment

TIME_STEPS=50
#file_path = findfile('Output.csv', '../data/real-time/')
file_path = '../data/real-time/Output.csv'
scaler = load_scaler('lstm-scaler.sav')

#### Find Files for Training/Testing ####

model = tf.keras.models.load_model('lstm-model.keras')

threshold = threshold_load('threshold.txt')


filecheck = os.stat(file_path).st_mtime
filetemp = filecheck
rows = 501

try:  
    while True:
        if filetemp != filecheck:
            
            X_test, y_test, rows = load_file_deploy(file_path, TIME_STEPS, rows, scaler)
            X_test_pred = model.predict(X_test, verbose=0)
            test_mae_loss = np.abs(X_test_pred-y_test)

            plot_deployment(test_mae_loss, X_test_pred.shape[1], threshold)
                            

        time.sleep(0.5)
        filecheck = os.stat(file_path).st_mtime
except KeyboardInterrupt:        
    print("End Deployment...")
