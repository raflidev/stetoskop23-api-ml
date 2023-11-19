#!/usr/bin/env-python

import warnings
import plotly.graph_objects as go
import json
import wave
import os
import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv
import sys
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import freqz
np.set_printoptions(threshold=sys.maxsize, suppress=True)
warnings.simplefilter("ignore", DeprecationWarning)

header = 'filename'
for i in range(1, 41):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

def clean_noise(audiofile):
    #music file, Fs
    song_sample, sampling_freq = sf.read(audiofile)
    #L
    filter_len = 101
    #Fc origigal 7500, kita coba 1000, 2000, dsb
    low_cutoff_freq = 7500
    zero_index = [0]
    #Ft
    normalized_transition_freq_low = low_cutoff_freq/sampling_freq
    #setting the initial weights filled with zeros in the array
    filter_coefficients = filter_len * zero_index
    #M
    filter_order = filter_len - 1
    two_pi = 2*np.pi
    i=0
    w_num=[]
    w_den = []
    #taking half of the M value
    half_filter = filter_order/2
    two_pi = 2*np.pi

    #low-pass filter algorithm
    two_norm_freq_low = 2*normalized_transition_freq_low
    for index in range(filter_len):
        if(index != half_filter):
            #dividing the numerator and denominator for the equation
            w_num = np.sin(two_pi*normalized_transition_freq_low*(index-half_filter))
            w_den = np.pi*(index-half_filter)
            filter_coefficients[index] = w_num/w_den
        else:
            filter_coefficients[index] = two_norm_freq_low

    #variable for hamming window values
    hamming_win_weights = zero_index * filter_len

    eq1 = 0.54
    coeff = 0.46

    for i in range(filter_len):
        h_num = two_pi*i
        h_cos = np.cos(h_num/filter_order)
        hamming_win_weights[i] = (eq1 - coeff*(h_cos))


    hamming_window1 = []

    windowed_output = zero_index * filter_len
    for i in range(filter_len):
        windowed_output[i] = filter_coefficients[i] * hamming_win_weights[i]


    x,y = freqz(filter_coefficients,1)
    a,b = freqz(windowed_output,1)

    hamming_window2 = []

    cleanFile = np.convolve(windowed_output,song_sample)
    sf.write(audiofile,cleanFile,sampling_freq)


def extract_mfcc(audiofile):
    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    clean_noise(audiofile)
    y, sr = librosa.load(audiofile, mono=True, duration=3,
                         sr=8000, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(
        y=y, sr=8000, n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128)
    to_append = f'Signal'
    for e in mfcc:
        to_append += f' {np.mean(e.T, axis=0)}'
    file = open('data.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())

def predict(audio):
  dt_audio = audio
  model = 'model.tflite'
  mfccs = extract_mfcc(dt_audio)
  data = pd.read_csv('data.csv')
  data = data.drop(['filename'], axis=1)
  X = np.array(data.iloc[:, :-1], dtype=float)
  X = np.reshape(X, (X.shape[0], 40, 1, 1))

  # Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=model)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(X, dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])

  hasil = np.argmax(output_data)

  spf = wave.open(dt_audio, 'r')
#   print(spf.getsampwidth() * 8)
  signal = spf.readframes(-1)
  signal = np.fromstring(signal, dtype='int16')
  fs = spf.getframerate()
  Time = np.linspace(0, len(signal) / fs, num=len(signal))
  path_dir = {}
  path_dir['z'] = f' {hasil}'
  path_dir['y'] = f' {signal}'
  path_dir['x'] = f' {Time}'
  # print(json.dumps(path_dir, separators=(',', ':')))
  return json.dumps({"file": audio, "hasil": str(hasil)})

  # fig = go.Figure([go.Scatter(x=Time, y=signal)])
  # fig.show()
