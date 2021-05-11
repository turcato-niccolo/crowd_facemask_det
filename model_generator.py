import pylab as p
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob
from utils import *
import time


def one_hot(n_classes, y):
    return np.eye(n_classes)[y]


def augmentImages(set_x, generator, num_augment_imgs):
    """ set_x:      array of images to augment
    generator:  Keras ImageDataGenerator to generate images
    num_augment_imgs: number of images to generate for each image in the set
    """
    # set_x = np.asarray(set_x).astype('float32')
    tensor_set = tf.expand_dims(set_x, 3)
    out_set_x = np.copy(set_x)
    out_set_x = np.resize(out_set_x, (
        np.shape(out_set_x)[0] + np.shape(out_set_x)[0] * num_augment_imgs, np.shape(out_set_x)[1],
        np.shape(out_set_x)[2]))
    for i in range(len(set_x)):
        it = generator.flow(tf.expand_dims(tensor_set[i], 0), batch_size=1)
        for j in range(num_augment_imgs):
            # generate batch of images
            batch = it.next()
            out_set_x[np.shape(set_x)[0] + num_augment_imgs * i + j] = batch[0].astype('uint8')[:, :, 0]

    return out_set_x


def convert_images_bgr_gray(images_bgr):
    imgs_gray = []
    for k in range(len(images_bgr)):
        imgs_gray.append(cv2.cvtColor(images_bgr[k], cv2.COLOR_BGR2GRAY))
    return np.array(imgs_gray)


def resize_images(images, sizex, sizey):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (sizex, sizey))
    return np.array(images)


def augment_set(set_x, set_y, generator, num_augmented_images):
    tensor_set = tf.expand_dims(set_x, 3)
    out_set_x = set_x.copy()
    out_set_y = set_y.copy()
    out_set_x = np.resize(out_set_x, (
        np.shape(out_set_x)[0] + np.shape(out_set_x)[0] * num_augmented_images, np.shape(out_set_x)[1],
        np.shape(out_set_x)[2]))
    out_set_y = np.resize(out_set_y, (np.shape(out_set_y)[0] + np.shape(out_set_y)[0] * num_augmented_images), )
    for i in range(set_x.shape[0]):
        it = generator.flow(tf.expand_dims(tensor_set[i], 0), batch_size=1)
        for j in range(num_augmented_images):
            # generate batch of images
            batch = it.next()
            out_set_x[np.shape(set_x)[0] + num_augmented_images * i + j] = batch[0].astype('uint8')[:, :, 0]
            out_set_y[np.shape(set_x)[0] + num_augmented_images * i + j] = set_y[i]
    return out_set_x, out_set_y


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

# create a datagenerator
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=30, width_shift_range=0.15,
                                                       height_shift_range=0.15, horizontal_flip=True,
                                                       brightness_range=(1., 1.))

# load images
images_without_bgr = [cv2.imread(file) for file in glob.glob("DATASET/without_mask/*")]
images_without_bgr += [cv2.imread(file) for file in glob.glob("DATA/without_mask/*")]
numNo = len(images_without_bgr)
images_correct_bgr = [cv2.imread(file) for file in glob.glob("DATASET/with_mask/*")]
images_correct_bgr += [cv2.imread(file) for file in glob.glob("DATA/with_mask/*")]
numWith = len(images_correct_bgr)
images_incorrect_bgr = [cv2.imread(file) for file in glob.glob("DATASET/incorrect_mask/*")]
images_incorrect_bgr += [cv2.imread(file) for file in glob.glob("DATA/incorrect_mask/*")]
numIncorrect = len(images_incorrect_bgr)

max_images_class = max([numNo, numWith, numIncorrect])

images_without_gray = resize_images(images_without_bgr, 100, 100)
images_without_gray = convert_images_bgr_gray(images_without_gray)

images_incorrect_gray = resize_images(images_incorrect_bgr, 100, 100)
images_incorrect_gray = convert_images_bgr_gray(images_incorrect_gray)

images_correct_gray = resize_images(images_correct_bgr, 100, 100)
images_correct_gray = convert_images_bgr_gray(images_correct_gray)

images_without_gray = augmentImages(images_without_gray, datagen, int(np.ceil(max_images_class / numNo) * 10))

images_incorrect_gray = augmentImages(images_incorrect_gray, datagen, int(np.ceil(max_images_class / numIncorrect) * 10))

images_correct_gray = augmentImages(images_correct_gray, datagen, int(np.ceil(max_images_class / numWith) * 10))

images_gray = np.concatenate((images_incorrect_gray, images_correct_gray, images_without_gray), axis=0)

numNo = images_without_gray.shape[0]
numWith = images_correct_gray.shape[0]
numIncorrect = images_incorrect_gray.shape[0]
numImages = images_gray.shape[0]

# set label (0-incorrect, 1-with, 2-without)
y = np.zeros((numImages,))
y = np.round(y)
y = y.astype(int)
y[numIncorrect:numIncorrect + numWith] = 1
y[numIncorrect + numWith:numIncorrect + numWith + numNo] = 2

numImages = images_gray.shape[0]

# get a random permutation of the element
indexes = np.random.permutation(range(0, numImages))

# creation of train, validation and test set (on paper it use 10% for test, 20% of 90% as validation)
num_train = 3 * numImages // 5  # 60%
num_val = numImages // 5  # 20%
num_test = numImages // 5  # 20%

index_train = indexes[0:num_train]
index_val = indexes[num_train:num_train + num_val]
index_test = indexes[num_train + num_val:num_train + num_val + num_test]

x_train = np.take(images_gray, index_train, axis=0)
x_val = np.take(images_gray, index_val, axis=0)
x_test = np.take(images_gray, index_test, axis=0)
y_train = np.take(y, index_train, axis=0)
y_val = np.take(y, index_val, axis=0)
y_test = np.take(y, index_test, axis=0)

x_train = x_train
x_val = x_val
x_test = x_test

x_train = tf.expand_dims(x_train, 3)
y_train = one_hot(3, y_train)
y_train = tf.convert_to_tensor(y_train)

x_val = tf.expand_dims(x_val, 3)
y_val = one_hot(3, y_val)
y_val = tf.convert_to_tensor(y_val)

x_test = tf.expand_dims(x_test, 3)
y_test = one_hot(3, y_test)
y_test = tf.convert_to_tensor(y_test)

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
