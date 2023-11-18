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
np.set_printoptions(threshold=sys.maxsize, suppress=True)
warnings.simplefilter("ignore", DeprecationWarning)

header = 'filename'
for i in range(1, 41):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


def extract_mfcc(audiofile):
    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
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
