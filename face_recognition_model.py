import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import voting
import operator

def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(tf.expand_dims(img, 0), 3)
    return img


class FacemaskRecognitionModel:
    def __init__(self, model_path_file):
        self.model = keras.models.load_model(model_path_file)

    def predict_one(self, image):
        image = preprocess_image(image)
        predicted_values = self.model(image)
        return np.argmax(predicted_values[0])


class FacemaskRecognitionModelFromEnsemble:
    def __init__(self, models_path_files):
        self.models = []
        for model_path_file in models_path_files:
            self.models.append(keras.models.load_model(model_path_file))

    def predict_one(self, image, voting_rule, n_classes, preprocess=True):
        if preprocess:
            image = preprocess_image(image)
        votes = []
        for model in self.models:
            votes.append(voting.one_hot2vote(model(image).numpy()[0]))
        election_results = voting.apply_voting(votes, voting_rule, range(n_classes))
        return max(election_results, key=election_results.get)

    def evaluate(self, x, y, voting_rule, n_classes):
        """
        :param x: Non preprocessed raw images
        :param y: labels
        :param voting_rule:
        :param n_classes:
        :return:
        """
        count_correct = 0
        for i in range(len(y)):
            label = self.predict_one(tf.expand_dims(x[i], 0), voting_rule, n_classes, preprocess=False)
            count_correct += 1 if label == np.argmax(y[i]) else 0
            print(str(i+1)+"/"+str(len(y)))

        return count_correct / len(y)
