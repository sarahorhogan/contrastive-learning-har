import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import SeparableConv1D, Conv1D, MaxPooling1D
#from tensorflow.keras.utils.vis_utils import plot_model
from simclr_models import create_base_model, attach_simclr_head

from raw_data_processing_UCI import get_train_test_data

import os
import pickle
import joblib
import numpy as np


def train_model(x_train, y_train, x_valid, y_valid):

    kernel_size = 32
    max_pool_size = 3
    dropout_rate = 0.5

    input_sample_size = x_train.shape[1:]
    
    model = Sequential()
    model.add(Conv1D(32, kernel_size, padding='same', input_shape=input_sample_size))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size= max_pool_size))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(16, kernel_size, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size= max_pool_size))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(6))
    model.add(Activation('softmax'))

    print(model.summary())

    # Compiling the model
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    batch_size = 64
    epochs = 40

    # Training the model
    model.fit(x_train,
          y_train,
          batch_size=batch_size,
          validation_data=(x_valid, y_valid),
          epochs=epochs)

    return model


if __name__ == "__main__":

    x_train, x_valid, y_train, y_valid, x_test, Y_test = get_train_test_data()

    '''
    model = train_model(x_train, y_train, x_valid, y_valid)

    model_json = model.to_json()

    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('model_h5',save_format='h5') #Will only save the weights so that you are able to apply them on a different architecture 

    model.save('Baseline_CNN') #Saves the whole architecture, weights and the optimizer state - the details needed to reconstitute your model
    '''

    #baseline_model = tensorflow.keras.models.load_model('Baseline_CNN')


