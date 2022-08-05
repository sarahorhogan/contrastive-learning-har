import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import pandas as pd
import os
import io 

from config import DATASET_PATH, INPUT_SIGNAL_TYPES, LABELS, TRAIN, TEST, X_train_signals_paths, X_test_signals_paths, y_train_path, y_test_path

'''
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import SeparableConv1D, Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model
'''

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = pd.read_csv(signal_type_path, sep='\s+', header=None)

        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(file)

    return X_signals

def load_y(y_path):
    file = pd.read_csv(y_path, sep='\s+', header=None).values
    # Read dataset from disk, dealing with text file's syntax

    # Substract 1 to each output class for friendly 0-based indexing
    return file


def label_mapping ():
    label_info = pd.read_csv(DATASET_PATH + "activity_labels.txt", sep='\s+', header=None)
    label_mapping = dict()

    for index, row_data in label_info.iterrows():
        label_mapping.update({row_data[0] -1 : row_data[1]})

    return label_mapping

def count_classes(y):
    return len(set([tuple(category) for category in y]))

def get_train_test_data():

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    x_train = np.stack(X_train, axis=-1)
    x_test = np.stack(X_test, axis=-1)

    Y_train = load_y(y_train_path)
    Y_test = load_y(y_test_path)

    encoder = OneHotEncoder(categories='auto')
    y_train = encoder.fit_transform(Y_train).toarray()

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, 
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=123)
    return x_train, x_valid, y_train, y_valid, x_test, Y_test
    











