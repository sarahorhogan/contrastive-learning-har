import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import SeparableConv1D, Conv1D, MaxPooling1D
#from tensorflow.keras.utils.vis_utils import plot_model
from simclr_models import create_base_model, attach_simclr_head
from simclr_utitlities import generate_composite_transform_function_simple, simclr_train_model
from raw_data_processing_UCI import get_train_test_data

import transformations

import os
import pickle
import joblib
import numpy as np


if __name__ == "__main__":

    x_train, x_valid, y_train, y_valid, x_test, Y_test = get_train_test_data()
    tensorflow.keras.backend.set_floatx('float32')

    base_model = create_base_model(x_train.shape[1:], model_name = "base_model")
    simclr_model = attach_simclr_head(base_model)
    print(simclr_model.summary())


    transform_funcs_vectorized = [
     transformations.scaling_transform_vectorized, # Use Scaling trasnformation
    ]

    transform_function = generate_composite_transform_function_simple(transform_funcs_vectorized)

    batch_size = 512
    decay_steps = 1000
    epochs = 200
    temperature = 0.1
    #lr_decayed_fn = tensorflow.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
    #optimizer = tensorflow.keras.optimizers.SGD(lr_decayed_fn)

    optimizer = tensorflow.keras.optimizers.SGD()

    np_train = (x_train, y_train)

    trained_simclr_model, epoch_losses = simclr_train_model(simclr_model, np_train[0], optimizer, batch_size, transform_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

    trained_simclr_model.save("simclr_model")

