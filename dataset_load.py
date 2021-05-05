import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob

def load_dataset(mult_augmentation=10):
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

    images_without_gray = augmentImages(images_without_gray, datagen, int(np.ceil(max_images_class / numNo) * mult_augmentation))

    images_incorrect_gray = augmentImages(images_incorrect_gray, datagen,
                                          int(np.ceil(max_images_class / numIncorrect) * mult_augmentation))

    images_correct_gray = augmentImages(images_correct_gray, datagen, int(np.ceil(max_images_class / numWith) * mult_augmentation))

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

    xtrain = np.take(images_gray, index_train, axis=0)
    xval = np.take(images_gray, index_val, axis=0)
    xtest = np.take(images_gray, index_test, axis=0)
    ytrain = np.take(y, index_train, axis=0)
    yval = np.take(y, index_val, axis=0)
    ytest = np.take(y, index_test, axis=0)

    xtrain = xtrain
    xval = xval
    xtest = xtest

    xtrain = tf.expand_dims(xtrain, 3)
    ytrain = one_hot(3, ytrain)
    ytrain = tf.convert_to_tensor(ytrain)

    xval = tf.expand_dims(xval, 3)
    yval = one_hot(3, yval)
    yval = tf.convert_to_tensor(yval)

    xtest = tf.expand_dims(xtest, 3)
    ytest = one_hot(3, ytest)
    ytest = tf.convert_to_tensor(ytest)

    return xtrain, ytrain, xval, yval, xtest, ytest

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