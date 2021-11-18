#!/usr/bin/env python
# coding: utf-8


import os
import random
import sys


## Package
import glob 
import keras
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow
py.init_notebook_mode(connected=True)


## Keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical


## Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


## Rest
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm


from sklearn.model_selection import StratifiedShuffleSplit
from keras import backend as K
import json
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#import pyaudio
import wave

def initialize():

    input_duration=3
    # % pylab inline

    # ## Reading the Data

    # Data Directory
    # Please edit according to your directory change.
    dir_list = os.listdir(r'SER\RAV')
    dir_list.sort()

    # Create DataFrame for Data intel
    data_df = pd.DataFrame(columns=['path', 'source', 'actor', 'gender',
                                    'intensity', 'statement', 'repetition', 'emotion'])
    count = 0
    for i in dir_list:
        file_list = os.listdir('SER\\RAV\\' + i)
        for f in file_list:
            nm = f.split('.')[0].split('-')
            path = r'SER\\RAV\\' + i + '\\' + f
            src = int(nm[1])
            actor = int(nm[-1])
            emotion = int(nm[2])
            
            if int(actor)%2 == 0:
                gender = "female"
            else:
                gender = "male"
            
            if nm[3] == '01':
                intensity = 0
            else:
                intensity = 1
            
            if nm[4] == '01':
                statement = 0
            else:
                statement = 1
            
            if nm[5] == '01':
                repeat = 0
            else:
                repeat = 1
                
            data_df.loc[count] = [path, src, actor, gender, intensity, statement, repeat, emotion]
            count += 1

    data_df.head(25)


    # ## Plotting the audio file's waveform and its spectrogram

    filename = data_df.path[1021]

    samples, sample_rate = librosa.load(filename)
    sample_rate, samples

    len(samples), sample_rate

    def log_specgram(audio, sample_rate, window_size=20,
                    step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)


    sample_rate/ len(samples)

    # Plotting Wave Form and Spectrogram
    freqs, times, spectrogram = log_specgram(samples, sample_rate)

    mean = np.mean(spectrogram, axis=0)
    std = np.std(spectrogram, axis=0)
    spectrogram = (spectrogram - mean) / std


    # Trim the silence voice
    aa , bb = librosa.effects.trim(samples, top_db=30)
    aa, bb


    # Original Sound
    ipd.Audio(samples, rate=sample_rate)


    # Silence trimmed Sound by librosa.effects.trim()
    ipd.Audio(aa, rate=sample_rate)


    # Silence trimmed Sound by manual trimming
    samples_cut = samples[10000:-12500]
    ipd.Audio(samples_cut, rate=sample_rate)


    # ## Defining the label

    # 2 class: Positive & Negative

    # Positive: Calm, Happy
    # Negative: Angry, Fearful, Sad

    label2_list = []
    for i in range(len(data_df)):
        if data_df.emotion[i] == 2: # Calm
            lb = "_positive"
        elif data_df.emotion[i] == 3: # Happy
            lb = "_positive"
        elif data_df.emotion[i] == 4: # Sad
            lb = "_negative"
        elif data_df.emotion[i] == 5: # Angry
            lb = "_negative"
        elif data_df.emotion[i] == 6: # Fearful
            lb = "_negative"
        else:
            lb = "_none"
            
        # Add gender to the label    
        label2_list.append(data_df.gender[i] + lb)
        
    len(label2_list)


    #3 class: Positive, Neutral & Negative

    # Positive:  Happy
    # Negative: Angry, Fearful, Sad
    # Neutral: Calm, Neutral

    label3_list = []
    for i in range(len(data_df)):
        if data_df.emotion[i] == 1: # Neutral
            lb = "_neutral"
        elif data_df.emotion[i] == 2: # Calm
            lb = "_neutral"
        elif data_df.emotion[i] == 3: # Happy
            lb = "_positive"
        elif data_df.emotion[i] == 4: # Sad
            lb = "_negative"
        elif data_df.emotion[i] == 5: # Angry
            lb = "_negative"
        elif data_df.emotion[i] == 6: # Fearful
            lb = "_negative"
        else:
            lb = "_none"
        
        # Add gender to the label  
        label3_list.append(data_df.gender[i] + lb)
        
    len(label3_list)


    # 5 class: angry, calm, sad, happy & fearful
    label5_list = []
    for i in range(len(data_df)):
        if data_df.emotion[i] == 2:
            lb = "_calm"
        elif data_df.emotion[i] == 3:
            lb = "_happy"
        elif data_df.emotion[i] == 4:
            lb = "_sad"
        elif data_df.emotion[i] == 5:
            lb = "_angry"
        elif data_df.emotion[i] == 6:
            lb = "_fearful"    
        else:
            lb = "_none"
        
        # Add gender to the label  
        label5_list.append(data_df.gender[i] + lb)
        
    len(label5_list)


    # All class

    label8_list = []
    for i in range(len(data_df)):
        if data_df.emotion[i] == 1:
            lb = "_neutral"
        elif data_df.emotion[i] == 2:
            lb = "_calm"
        elif data_df.emotion[i] == 3:
            lb = "_happy"
        elif data_df.emotion[i] == 4:
            lb = "_sad"
        elif data_df.emotion[i] == 5:
            lb = "_angry"
        elif data_df.emotion[i] == 6:
            lb = "_fearful"
        elif data_df.emotion[i] == 7:
            lb = "_disgust"
        elif data_df.emotion[i] == 8:
            lb = "_surprised"
        else:
            lb = "_none"
            
        # Add gender to the label 
        label8_list.append(data_df.gender[i]  + lb)
        
    len(label8_list)


    # Select the label set you want by commenting the unwanteds.

    data_df['label'] = label5_list
    data_df.head()


    # Plotting the emotion distribution

    def plot_emotion_dist(dist, color_code='#C2185B', title="Plot"):
        """
        To plot the data distributioin by class.
        Arg:
        dist: pandas series of label count. 
        """
        tmp_df = pd.DataFrame()
        tmp_df['Emotion'] = list(dist.keys())
        tmp_df['Count'] = list(dist)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax = sns.barplot(x="Emotion", y='Count', color=color_code, data=tmp_df)
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)


    # ## Data Splitting

    # Male Data Set

    data2_df = data_df.copy()
    data2_df = data2_df[data2_df.label != "male_none"]
    data2_df = data2_df[data2_df.label != "female_none"].reset_index(drop=True)
    data2_df = data2_df[data2_df.label != "female_neutral"]
    data2_df = data2_df[data2_df.label != "female_happy"]
    data2_df = data2_df[data2_df.label != "female_angry"]
    data2_df = data2_df[data2_df.label != "female_sad"]
    data2_df = data2_df[data2_df.label != "female_fearful"]
    data2_df = data2_df[data2_df.label != "female_calm"]
    data2_df = data2_df[data2_df.label != "female_positive"]
    data2_df = data2_df[data2_df.label != "female_negative"].reset_index(drop=True)

    tmp1 = data2_df[data2_df.actor == 21]
    tmp2 = data2_df[data2_df.actor == 22]
    tmp3 = data2_df[data2_df.actor == 23]
    tmp4 = data2_df[data2_df.actor == 24]
    data3_df = pd.concat([tmp1, tmp3],ignore_index=True).reset_index(drop=True)
    data2_df = data2_df[data2_df.actor != 21]
    data2_df = data2_df[data2_df.actor != 22]
    data2_df = data2_df[data2_df.actor != 23].reset_index(drop=True)
    data2_df = data2_df[data2_df.actor != 24].reset_index(drop=True)
    data2_df.head(50)

    data3_df.head(80)


    # ## Analysing Features of audio files using librosa


    data = pd.DataFrame(columns=['feature'])
    for i in tqdm(range(len(data2_df))):
        X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        data.loc[i] = [feature]

    data.head()

    df3 = pd.DataFrame(data['feature'].values.tolist())
    labels = data2_df.label

    df3.head()

    newdf = pd.concat([df3,labels], axis=1)

    rnewdf = newdf.rename(index=str, columns={"0": "label"})
    len(rnewdf)

    rnewdf.head(10)

    rnewdf.isnull().sum().sum()

    rnewdf = rnewdf.fillna(0)
    rnewdf.head()


    # ## Data Making/Processing


    def plot_time_series(data):
        """
        Plot the Audio Frequency.
        """
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()


    def noise(data):
        """
        Adding White Noise.
        """
        # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
        noise_amp = 0.005*np.random.uniform()*np.amax(data)
        data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
        return data
        
    def shift(data):
        """
        Random Shifting.
        """
        s_range = int(np.random.uniform(low=-5, high = 5)*500)
        return np.roll(data, s_range)
        
    def stretch(data, rate=0.8):
        """
        Streching the Sound.
        """
        data = librosa.effects.time_stretch(data, rate)
        return data
        
    def pitch(data, sample_rate):
        """
        Pitch Tuning.
        """
        bins_per_octave = 12
        pitch_pm = 2
        pitch_change =  pitch_pm * 2*(np.random.uniform())   
        data = librosa.effects.pitch_shift(data.astype('float64'), 
                                        sample_rate, n_steps=pitch_change, 
                                        bins_per_octave=bins_per_octave)
        return data
        
    def dyn_change(data):
        """
        Random Value Change.
        """
        dyn_change = np.random.uniform(low=1.5,high=3)
        return (data * dyn_change)
        
    def speedNpitch(data):
        """
        peed and Pitch Tuning.
        """
        # you can change low and high here
        length_change = np.random.uniform(low=0.8, high = 1)
        speed_fac = 1.0  / length_change
        tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
        minlen = min(data.shape[0], tmp.shape[0])
        data *= 0
        data[0:minlen] = tmp[0:minlen]
        return data


    # Data Making Method 1

    syn_data1 = pd.DataFrame(columns=['feature', 'label'])
    for i in tqdm(range(len(data2_df))):
        X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
        if data2_df.label[i]:
            X = noise(X)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            a = random.uniform(0, 1)
            syn_data1.loc[i] = [feature, data2_df.label[i]]


    # Data Making Method 2

    syn_data2 = pd.DataFrame(columns=['feature', 'label'])
    for i in tqdm(range(len(data2_df))):
        X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
        if data2_df.label[i]:
            X = pitch(X, sample_rate)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
            feature = mfccs
            a = random.uniform(0, 1)
            syn_data2.loc[i] = [feature, data2_df.label[i]]


    len(syn_data1), len(syn_data2)

    syn_data1 = syn_data1.reset_index(drop=True)
    syn_data2 = syn_data2.reset_index(drop=True)


    df4 = pd.DataFrame(syn_data1['feature'].values.tolist())
    labels4 = syn_data1.label
    syndf1 = pd.concat([df4,labels4], axis=1)
    syndf1 = syndf1.rename(index=str, columns={"0": "label"})
    syndf1 = syndf1.fillna(0)
    len(syndf1)

    syndf1.head()

    df4 = pd.DataFrame(syn_data2['feature'].values.tolist())
    labels4 = syn_data2.label
    syndf2 = pd.concat([df4,labels4], axis=1)
    syndf2 = syndf2.rename(index=str, columns={"0": "label"})
    syndf2 = syndf2.fillna(0)
    len(syndf2)

    syndf2.head()

    # Combining the Proccessed data with original
    combined_df = pd.concat([rnewdf, syndf1, syndf2], ignore_index=True)
    combined_df = combined_df.fillna(0)
    combined_df.head()


    #  Stratified Shuffle Split
    from sklearn.model_selection import StratifiedShuffleSplit
    X = combined_df.drop(['label'], axis=1)
    y = combined_df.label
    xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
    for train_index, test_index in xxx.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    y_train.value_counts()

    y_test.value_counts()

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    y_train

    X_train

    X_train.shape


    # ## Creating the CNN Model

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)


    # Set up Keras util functions

    from keras import backend as K

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def fscore(y_true, y_pred):
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        f_score = 2 * (p * r) / (p + r + K.epsilon())
        return f_score

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr


    # New model
    model = Sequential()
    model.add(Conv1D(256, 8, padding='valid',input_shape=(X_train.shape[1],1)))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    # Edit according to target class no.
    model.add(Dense(5))
    model.add(Activation('softmax'))
    #opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

    model.summary()

    # Compile your model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Model Training

    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
    # Please change the model name accordingly.
    mcp_save = ModelCheckpoint('Data_noiseNshift.h5', save_best_only=True, monitor='val_loss', mode='min')
    cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=100,
                        validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce])


    # Saving the model.json

    import json
    model_json = model.to_json()
    with open("SER\\Predictions Data\\model.json", "w") as json_file:
        json_file.write(model_json)


    import tensorflow as tf
    from tensorflow.keras.initializers import glorot_uniform
    loaded_model = tf.keras.models.load_model("Data_noiseNshift.h5",custom_objects={'GlorotUniform': glorot_uniform()})
    
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)


    # ## Predicting emotions on the test data

    len(data2_df)

    data_test = pd.DataFrame(columns=['feature'])
    for i in tqdm(range(len(data2_df))):
        X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        feature = mfccs
        data_test.loc[i] = [feature]
        
    test_valid = pd.DataFrame(data_test['feature'].values.tolist())
    test_valid = np.array(test_valid)
    test_valid_lb = np.array(data2_df.label)
    lb = LabelEncoder()
    test_valid_lb = np_utils.to_categorical(lb.fit_transform(test_valid_lb))
    test_valid = np.expand_dims(test_valid, axis=2)

    preds = loaded_model.predict(test_valid, 
                            batch_size=16, 
                            verbose=1)

    preds

    preds1=preds.argmax(axis=1)

    preds1

    abc = preds1.astype(int).flatten()

    predictions = (lb.inverse_transform((abc)))

    preddf = pd.DataFrame({'predictedvalues': predictions})
    preddf[:10]

    actual=test_valid_lb.argmax(axis=1)
    abc123 = actual.astype(int).flatten()
    actualvalues = (lb.inverse_transform((abc123)))

    actualdf = pd.DataFrame({'actualvalues': actualvalues})
    actualdf[:10]


    finaldf = actualdf.join(preddf)

    # ## Actual vs Predicted Values

    finaldf[40:60]

    finaldf.groupby('actualvalues').count()

    finaldf.groupby('predictedvalues').count()

    finaldf.to_csv('SER\\Predictions Data\\Predictions.csv', index=False)


    from sklearn.metrics import accuracy_score
    y_true = finaldf.actualvalues
    y_pred = finaldf.predictedvalues
    accuracy_score(y_true, y_pred)*100


    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix
    c = confusion_matrix(y_true, y_pred)
    c

    # Visualize Confusion Matrix 

    class_names = ['male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']
    
    return model, lb



def analyze(model, lb, file):
    data, sampling_rate = librosa.load("./"+file)
    ipd.Audio("./"+file)

    X, sample_rate = librosa.load("./"+file
                                ,res_type='kaiser_fast'
                                ,duration=3
                                ,sr=44100
                                ,offset=0.5)


    sample_rate = np.array(sample_rate)
    mfcc_test = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40),axis=0)
    mfcc_test = pd.DataFrame(data=mfcc_test).T
    mfcc_test

    mfcc_test= np.expand_dims(mfcc_test, axis=2)
    pred_test = model.predict(mfcc_test, 
                            batch_size=16, 
                            verbose=0)

    pred_test

    result = pred_test.argmax(axis=1)
    result = result.astype(int).flatten()
    result = (lb.inverse_transform((result)))
    result
    return result
