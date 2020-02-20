import glob
import cv2
import numpy as np


import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Conv2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras import losses, optimizers, regularizers

NO_LIGHT_DATA_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/no_light/'
RED_LIGHT_DATA_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/has_light/red/'
YELLOW_LIGHT_DATA_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/has_light/yellow/'
GREEN_LIGHT_DATA_PATH = '/home/workspace/teamwork/dataset/simulator_0220_3/has_light/green/'


def import_data_set():
    training_data = []
    for name in glob.glob(RED_LIGHT_DATA_PATH + '*.jpg'):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (60, 80))
        training_data.append((resized_img / 255., '0'))
    for name in glob.glob(YELLOW_LIGHT_DATA_PATH + '*.jpg'):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (60, 80))
        training_data.append((resized_img / 255., '1'))
#         training_data.append((resized_img / 255., '1'))
    for name in glob.glob(GREEN_LIGHT_DATA_PATH + '*.jpg'):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (60, 80))
        training_data.append((resized_img / 255., '2'))
#         training_data.append((resized_img / 255., '2'))
# too many no_light images, use one-third of them
    idx = 0
    for name in glob.glob(NO_LIGHT_DATA_PATH + '*.jpg'):
        if idx % 3 == 0:
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img, (60, 80))
            training_data.append((resized_img / 255., '3'))
        idx += 1
    return training_data


def generate_training_validation_data(dataset):
    X_data = []
    y_data = []
    for data in dataset:
        X_data.append(data[0])
        y_data.append(data[1])
    return np.array(X_data), np.array(y_data)


data_set = import_data_set()

print("training data size: {}".format(len(data_set)))
print("first image size: ")
print(data_set[0][0].shape)
data = generate_training_validation_data(data_set)

categorical_labels = to_categorical(data[1])

num_classes = 4
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(80, 60, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(2,2))
Dropout(0.8)

# model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
# model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(2,2))
Dropout(0.8)
model.add(Flatten())

#model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dense(16, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(num_classes, activation='softmax'))


loss = losses.categorical_crossentropy
optimizer = optimizers.Adam()

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

model.fit(data[0], categorical_labels, batch_size=32, epochs=25, verbose=True, validation_split=0.15, shuffle=True)

score = model.evaluate(data[0], categorical_labels, verbose=0)
print(score)

model.save('tl_classifier_simulator.h5')
