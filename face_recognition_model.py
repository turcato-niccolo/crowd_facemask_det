import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2


class FacemaskRecognitionModel:
    def __init__(self, model_path_file):
        self.model = keras.models.load_model(model_path_file)

    def predict_one(self, image):
        image = cv2.resize(image, (100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(tf.expand_dims(image, 0), 3)
        predicted_values = self.model(image)
        return np.argmax(predicted_values[0])
