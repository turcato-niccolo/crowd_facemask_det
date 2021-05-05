import pylab as p
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob
import dataset_load
from utils import *
import time
from gridSearch_template import *

(x_train, y_train, x_val, y_val, x_test, y_test) = dataset_load.load_dataset(2)

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
param_distribs = {
    "learning_rate": [1e-3, 4e-3, 7e-3, 1e-4, 4e-4, 7e-4],
    "alpha_l1_value": [4e-3, 7e-3, 1e-4, 4e-4, 7e-4, 1e-5, 1e-6],
    "alpha_l2_value": [4e-3, 7e-3, 1e-4, 4e-4, 7e-4, 1e-5],
    "dropout": [0.5, 0.4, 0.3, 0.2, 0.1],
    "kernel1_size": [[3, 3], [4, 4], [5, 5]],
    "kernel2_size": [[3, 3], [4, 4]],
    "activation1": ["relu", "sigmoid"],
    "activation2": ["relu", "sigmoid"]
}

grid_search = GridSearchCV(keras_reg, param_distribs)
grid_search.fit(x_train.numpy(), y_train.numpy(), epochs=20, validation_data=(x_val.numpy(), y_val.numpy()))

print(grid_search.best_params_)

model = grid_search.best_estimator_.model

scores = model.evaluate(x_test, y_test, verbose=2)
print(" %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("----------------------------")

# save model and architecture to single file
model.save("models/facemask_model.h5")
