import pylab as p
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob
from utils import *

def augment_set(set_x,set_y, generator, num_augmented_images):
    tensor_set = tf.expand_dims(set_x, 3)
    out_set_x = set_x.copy()
    out_set_y = set_y.copy()
    out_set_x = np.resize(out_set_x, (np.shape(out_set_x)[0]+np.shape(out_set_x)[0]*num_augmented_images,
                                      np.shape(out_set_x)[1], np.shape(out_set_x)[2]))
    out_set_y = np.resize(out_set_y, (np.shape(out_set_y)[0]+np.shape(out_set_y)[0]*num_augmented_images),)
    for i in range(set_x.shape[0]):
        it = generator.flow(tf.expand_dims(tensor_set[i], 0), batch_size=1)
        for j in range(num_augmented_images):
            # generate batch of images
            batch = it.next()
            out_set_x[np.shape(set_x)[0]+num_augmented_images*i+j] = batch[0].astype('uint8')[:,:,0]
            out_set_y[np.shape(set_x)[0]+num_augmented_images*i+j] = set_y[i]
    return out_set_x, out_set_y

def one_hot(n_classes, y):
    return np.eye(n_classes)[y]

np.random.seed(42)

#create a datagenerator
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=20,width_shift_range=0.15,
                                                       height_shift_range=0.15, horizontal_flip=True)

# load images
images_without_bgr = [cv2.imread(file) for file in glob.glob("train/without_mask/*.jpg")]
numNo = len(images_without_bgr)
images_correct_bgr = [cv2.imread(file) for file in glob.glob("train/with_mask/*.jpg")]
numWith = len(images_correct_bgr)
images_incorrect_bgr = [cv2.imread(file) for file in glob.glob("train/mask_weared_incorrect/*.jpg")]
numIncorrect = len(images_incorrect_bgr)

images_bgr = images_incorrect_bgr + images_correct_bgr + images_without_bgr
numImages = len(images_bgr)

images_gray = list()

# resize
for i in range(numImages):
    images_bgr[i] = cv2.resize(images_bgr[i], (100, 100))

# convert into gray-scale
for i in range(numImages):
    images_gray.append(cv2.cvtColor(images_bgr[i], cv2.COLOR_BGR2GRAY))

# set label (0-incorrect, 1-with, 2-without)
y = np.zeros((numImages,))
y = np.round(y)
y = y.astype(int)
y[numIncorrect:numIncorrect + numWith] = 1
y[numIncorrect + numWith:numIncorrect + numWith + numNo] = 2

images_gray2 = np.copy(images_gray)
images_gray2, y = augment_set(images_gray2,y,datagen,10)
numImages = images_gray2.shape[0]

# get a random permutation of the element
indexes = np.random.permutation(range(0, numImages))

# creation of train, validation and test set (on paper it use 10% for test, 20% of 90% as validation)
num_train = 3 * numImages // 5  # 60%
num_val = numImages // 5  # 20%
num_test = numImages // 5  # 20%

index_train = indexes[0:num_train]
index_val = indexes[num_train:num_train + num_val]
index_test = indexes[num_train + num_val:num_train + num_val + num_test]

x_train = np.take(images_gray2, index_train, axis=0)
x_val = np.take(images_gray2, index_val, axis=0)
x_test = np.take(images_gray2, index_test, axis=0)
y_train = np.take(y, index_train, axis=0)
y_val = np.take(y, index_val, axis=0)
y_test = np.take(y, index_test, axis=0)

# normalization
mean_train = np.mean(x_train, axis=0)
std_train = np.std(x_train, axis=0)
x_train_scaled = (x_train - mean_train) / std_train
x_val_scaled = (x_val - mean_train) / std_train
x_test_scaled = (x_test - mean_train) / std_train

x_train_scaled = tf.expand_dims(x_train_scaled, 3)
y_train = one_hot(3, y_train)
y_train = tf.convert_to_tensor(y_train)

x_val_scaled = tf.expand_dims(x_val_scaled, 3)
y_val = one_hot(3, y_val)
y_val = tf.convert_to_tensor(y_val)

x_test_scaled = tf.expand_dims(x_test_scaled, 3)
y_test = one_hot(3, y_test)
y_test = tf.convert_to_tensor(y_test)

# create cnn
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=200, kernel_size=[3, 3], padding="valid", activation="relu", input_shape=[100, 100, 1]),
    keras.layers.MaxPool2D(pool_size=[3, 3]),
    keras.layers.Conv2D(filters=100, kernel_size=[3, 3], activation="relu"),
    keras.layers.MaxPool2D(pool_size=[3, 3]),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=["accuracy"])

# batch not given
history = model.fit(x_train_scaled, y_train, epochs=20, batch_size=50,
                    validation_data=(x_val_scaled, y_val))

# save model and architecture to single file
model.save("model.h5")

## load model
# model = keras.models.load_model('model.h5')

scores = model.evaluate(x_test_scaled, y_test, verbose=2)
print(" %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("----------------------------")

print(0)
