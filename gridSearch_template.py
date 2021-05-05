from matplotlib import pylab as p
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob
from utils import *
import time
from sklearn.model_selection import GridSearchCV  # needed


# here you put every possible parameter that you want to be processed into the gridsearch e.g alpha = 30 (i don't know if
# the initialization is needed)
def build_model(alpha_l1_value=0.001, alpha_l2_value=0.001, learning_rate=1, dropout=0.5, kernel1_size=[4, 4],
                kernel2_size=[3, 3], activation1="relu", activation2="relu"):
    # definition of the model. If it has some parameters passed into the function
    # as argument use it
    # e.g:
    model = keras.models.Sequential([
        keras.layers.BatchNormalization(input_shape=[100, 100, 1]),
        keras.layers.Conv2D(filters=200, kernel_size=kernel1_size, padding="valid", activation=activation1,
                            kernel_regularizer=tf.keras.regularizers.L2(alpha_l2_value)),
        keras.layers.MaxPool2D(pool_size=[3, 3]),
        keras.layers.Conv2D(filters=100, kernel_size=kernel2_size, activation=activation2,
                            kernel_regularizer=tf.keras.regularizers.L2(alpha_l2_value)),
        keras.layers.MaxPool2D(pool_size=[3, 3]),
        keras.layers.Flatten(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(alpha_l1_value)),
        keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.L1(alpha_l1_value))
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    # the model must be compiled
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    # the model must be returned
    return model


# into the code we must define a keras wrapper and the list of parameters with
# the values on wich the gridsearch will work
# e.g
'''
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
param_distribs = {
    "learning_rate": [1e-3, 4e-3, 7e-3, 1e-4, 4e-4, 7e-4],
    "alpha_l1_value": [4e-3, 7e-3, 1e-4, 4e-4, 7e-4, 1e-5],
    "alpha_l2_value": [4e-3, 7e-3, 1e-4, 4e-4, 7e-4, 1e-5],
    "dropout": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
}
'''

'''
    # we have to create the grid search object
    grid_search = GridSearchCV(keras_reg, param_distribs)
    
    # to start the gridSearch use fit
    # if the data passed are tensor, they must me converted using the method of
    # tensorflow numpy()
    # e.g:
    grid_search.fit(x_train_scaled.numpy(), y_train.numpy(), epochs=35,
                    validation_data=(x_val_scaled.numpy(), y_val.numpy()))
    
    # the parameters of the best model will be stored into the grid search object and can be retrieved
    # using grid_search.best_params_
    
    print(grid_search.best_params_)
    
    # to obtain the best model use grid_search.best_estimator_.model
    model = grid_search.best_estimator_.model


'''