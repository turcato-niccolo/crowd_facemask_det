import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.epoch, history.history['loss'])
    plt.plot(history.epoch, history.history['val_loss'])
    plt.title('loss')


def plot_accuracy(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.epoch, history.history['accuracy'])
    plt.plot(history.epoch, history.history['val_accuracy'])
    plt.title('accuracy')


def one_hot(n_classes, y):
    return np.eye(n_classes)[y]
