import pylab as p
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob
import dataset_load
from utils import *
import time


print("********************************")
np.random.seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

(x_train, y_train, x_val, y_val, x_test, y_test) = dataset_load.load_dataset()

alpha_l2 = 8e-3
alpha_l1 = 0
# create cnn
model = keras.models.Sequential([
    keras.layers.BatchNormalization(input_shape=[100, 100, 1]),
    keras.layers.Conv2D(filters=200, kernel_size=[4, 4], padding="valid", activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L2(alpha_l2)),
    keras.layers.MaxPool2D(pool_size=[3, 3]),
    keras.layers.Conv2D(filters=100, kernel_size=[3, 3], activation="relu",
                        kernel_regularizer=tf.keras.regularizers.L2(alpha_l2)),
    keras.layers.MaxPool2D(pool_size=[3, 3]),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(alpha_l1)),
    keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.L1(alpha_l1))
])

opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# history = model.fit(x_train_scaled, y_train, epochs=100, batch_size=50, validation_data=(x_val_scaled, y_val), callbacks=[callback])

# Fit data generator

datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=45, width_shift_range=0.15,
                                                       height_shift_range=0.15, horizontal_flip=True,
                                                       brightness_range=(1., 1.))
datagen.fit(x_train)
history = model.fit(datagen.flow(x_train, y_train, batch_size=100), epochs=100, validation_data=(x_val, y_val),
                    callbacks=[callback])

scores = model.evaluate(x_test, y_test, verbose=2)
print(" %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("----------------------------")

# save model and architecture to single file
model.save("models/facemask_model.h5")
