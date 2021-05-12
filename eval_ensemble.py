import glob

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import voting
import dataset_load
import face_recognition_model



(x_train, y_train, x_val, y_val, x_test, y_test) = dataset_load.load_dataset(1)
X = x_train
Y = y_train

facemask_ens = face_recognition_model.FacemaskRecognitionModelFromEnsemble(glob.glob("models/ensemble/ensemble*_model.h5"))

print("Accuracy (plurality): " + str(facemask_ens.evaluate(X, Y, voting.plurality_voting_rule, 3) * 100))

print("Accuracy (2-approval): " + str(facemask_ens.evaluate(X, Y, voting.approval2_voting_rule, 3) * 100))

print("Accuracy (borda): " + str(facemask_ens.evaluate(X, Y, voting.borda_voting_rule, 3) * 100))




