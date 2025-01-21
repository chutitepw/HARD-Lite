import os
import fnmatch
import pandas as pd
import numpy as np
import imblearn
import joblib
import statistics
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def findfile(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

#### Load training file ####
def load_file(file_path, window_size):
    sequenceX, sequenceY = [], []
    for files in file_path:
        df = pd.read_csv(files)
        #### Prep data sequence for each file ####:
        for i in range(0, len(df)-window_size):
            df_slice = df.iloc[i:i+window_size]
            sequenceX.append(df_slice)
            if df_slice.iloc[-1,-1] == 'ransom':
                sequenceY.append(1)
            else:
                sequenceY.append(0)
    print("Finsish loading file length: ",len(sequenceX))
    sequenceX, sequenceY = np.array(sequenceX), np.array(sequenceY)
    sequenceX = sequenceX[:,:, 0:sequenceX.shape[2]-1]
    
    return sequenceX, sequenceY

#### Load training file ####
def load_file_anomaly(file_path, window_size, training, scaler=None):
    dataset = []
    for files in file_path:
        print(files)
        df = pd.read_csv(files, usecols=range(0, pd.read_csv(files, nrows=1).shape[1] - 1), skiprows=1, header=None)

        #print(df.shape)
        #### Prep data sequence for each file ####:

        dataset.append(df)
    dataset = pd.concat(dataset)
    #print("Data Shape: ", dataset.shape)

    if training:
        scaler = create_scaler(dataset)
    dataset = scaler.transform(dataset)
    #print("Scaled Data Shape: ", dataset.shape)

    sequenceX, sequenceY = create_sequences(dataset, dataset[0], window_size)
    print("Finsish loading file shape: ",sequenceX.shape, sequenceY.shape)

    return sequenceX, sequenceY, scaler

#### Load deployment file ####
def load_file_deploy(file_path, window_size, rows, scaler=None):
    dataset = []
 
    df = pd.read_csv(file_path, usecols=range(0, pd.read_csv(file_path, nrows=1).shape[1] - 1), skiprows=rows-window_size, header=None)

    if(len(df)>=window_size):

        #### Prep data sequence for file ####:
        dataset = scaler.transform(df)
        #print("Scaled Data Shape: ", dataset.shape)

        rows += len(dataset)

        sequenceX, sequenceY = create_sequences(dataset, dataset[0], window_size)
        print("Finsish loading file shape: ",sequenceX.shape, sequenceY.shape)

    return sequenceX, sequenceY, rows

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X[i:(i+time_steps)])
        ys.append(X[i+time_steps])
    return np.array(Xs), np.array(ys)

def create_scaler(sequenceX):
    scaler = StandardScaler()
    scaler = scaler.fit(sequenceX)
    joblib.dump(scaler, 'lstm-scaler.sav')

    return scaler

def load_scaler(scalar_path):
    scaler = joblib.load(scalar_path)

    return scaler

#### Balance classes ####
def class_rebalance(sequenceX, sequenceY, window_size):
    sequenceX = sequenceX.reshape(sequenceX.shape[0],-1)
    rus = RandomUnderSampler(random_state=42, replacement=True)
    x_rus, y_rus = rus.fit_resample(sequenceX,sequenceY)
    x_rus = x_rus.reshape(x_rus.shape[0],window_size,x_rus.shape[1]//window_size)
    print("Classes rebalance: ",x_rus.shape,y_rus.shape)
    val, count = np.unique(y_rus, return_counts=True)
    # of classes
    for v, c in zip(val,count):
        print(f"{v}: {c}")

    return x_rus, y_rus

def threshold_calculation(train_mae_loss):
    threshold = [] 
    for j in range(0,train_mae_loss.shape[1]):
        scores = train_mae_loss[:,j]
        cut_off = statistics.mean(scores) + (2*statistics.pstdev(scores))
        threshold.append(cut_off)
    print(f'Reconstruction error threshold: {threshold}')
    return threshold

def threshold_save(threshold):
    f = open("threshold.txt", "w")
    for element in threshold:
        f.write("%s," %element)
    f.close()

def threshold_load(threshold_path):
    threshold = [] 
    f = open(threshold_path, "r")
    content = f.read()
    contentsplit = content.split(",")
    for j in range(0,len(contentsplit)-1):
        threshold.append(float(contentsplit[j]))
    print(f'Reconstruction error threshold: {threshold}')

    return threshold

def plot(history):
    if 'accuracy' in history.history.keys():
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_anomaly(test_mae_loss, num_evt, threshold):

    #### Plot prediction graph for each event ####
    plt.figure()
    for group in range(0,num_evt):
        plt.subplot(num_evt, 1, group+1)
        plt.ylabel("Prediction error")
        plt.ylim([0,10])
        plt.plot(test_mae_loss[:,group], label="Prediction")
        plt.axhline(y=threshold[group], color='r',label="Threshold")
        plt.legend()

    plt.xlabel("Time")
    
    plt.show()

def plot_deployment(test_mae_loss, num_evt, threshold):

    #### Plot prediction graph for each event ####
    plt.clf()
    for group in range(0,num_evt):
        plt.subplot(num_evt, 1, group+1)
        plt.ylabel("Prediction error")
        plt.axhline(y=threshold[group], color='r',label="Threshold")
        plt.plot(test_mae_loss[:,group], label="Prediction")
        plt.ylim([0,10])
        plt.legend()
    plt.xlabel("Sample")
    plt.draw()

    plt.pause(0.001)