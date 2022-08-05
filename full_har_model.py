import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import SeparableConv1D, Conv1D, MaxPooling1D
#from tensorflow.keras.utils.vis_utils import plot_model
from raw_data_processing_UCI import get_train_test_data

import simclr_models
import simclr_utitlities


if __name__ == "__main__":

    total_epochs = 5
    batch_size = 200
    tag = "full_eval"

    simclr_model = tensorflow.keras.models.load_model("simclr_model")
    full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(simclr_model, 6, model_name="TPN", intermediate_layer=7, last_freeze_layer=4)

    best_model_callback = tensorflow.keras.callbacks.ModelCheckpoint("full_evaluation",
        monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
    )

    x_train, x_valid, y_train, y_valid, x_test, Y_test = get_train_test_data()

    np_train = (x_train,y_train)
    np_val =(x_valid, y_valid)
    np_test = (x_test, Y_test)


    training_history = full_evaluation_model.fit(
        x = np_train[0],
        y = np_train[1],
        batch_size=batch_size,
        shuffle=True,
        epochs=total_epochs,
        callbacks=[best_model_callback],
        validation_data=np_val
    )
    print(simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True))
