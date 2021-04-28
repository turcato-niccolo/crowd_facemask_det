import tensorflow as tf
from tensorflow import keras
import numpy as np

class facemask_recognition_model:
    def __init__(self, model_path_file):
        self.model = keras.models.load_model('model.h5')

    def predict_one(self, image):
        predicted_values = self.model(image)
        return np.argmax(predicted_values[0])