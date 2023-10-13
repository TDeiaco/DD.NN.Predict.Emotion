import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
import time
import tensorflow as tf
import keras
import keras.optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.utils import class_weight
from keras import backend as K

#scale image to IMG_SIZE
IMG_SIZE = 48
SETS = ["test", "train", "val"]


#NOTE: If CATEGORIES changes, make sure to delete the .pickle data save files
#CATEGORIES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CATEGORIES = [ "happy", "sad", "neutral"]

#create one hot targets which represent each label, so there's len(CATEGORY) amount of one hots
targets = np.array([list(range(len(CATEGORIES)))]).reshape(-1)
one_hot_targets = np.eye(len(CATEGORIES))[targets]

#get lowest sample/image count within each category folder
def get_lowest_sample_count(set):
    lowestSampleCount = 10000000000
    for category in CATEGORIES:  # do each category
        path = os.path.join(f"data/{set}",f"{category}")  # create path to each category
        numSamples = len(os.listdir(path))
        if lowestSampleCount > numSamples: lowestSampleCount = numSamples

    return lowestSampleCount

#creates an array of X, y values, where X is the input img vector, and X are the classifications/labels
def process_data(set, balance_to_lowest=False):
    lowestSampleCount = 0
    if balance_to_lowest:
        lowestSampleCount = get_lowest_sample_count(set)

    data = []
    for category in CATEGORIES:  # do each category
        path = os.path.join(f"data/{set}",f"{category}")  # create path to each category
        loaded = 0
        for img in tqdm(os.listdir(path)):  # iterate over each image per each category
            if(balance_to_lowest):
                if(loaded >= lowestSampleCount):
                    break
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

                data.append([new_array, one_hot_targets[CATEGORIES.index(category)]])  # add this to our training_data
                loaded += 1
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    random.shuffle(data)
    return data

#hyperparamters
FILTER_SIZES = [3]
DENSE_SIZES = [1024]
DENSE_LAYERS = [1,2,3]
CONV_SIZE = [256]
EPOCHS = 50

time_label = int(time.time())
#now iterate through hyperparameters
for dense_layers in DENSE_LAYERS:
    for filter_size in FILTER_SIZES:
        for dense_size in DENSE_SIZES:
            for conv_size in CONV_SIZE:

                model_name = f"EmoteCNN-conv-{conv_size}-dens-{dense_size}-densLyr-{dense_layers}-fltr-{filter_size}-{time_label}"
                model_dir = f"models/{model_name}"
                checkpoint_filename = f"{model_dir}/model"

                #create dirs for checkpoint saving, dirs need to exist first
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                #the almighty data
                X = []
                y = []

                #load if doesn't exist
                try:
                    print("Loading training data.. \n\n")
                    pickle_in = open("X.pickle","rb")
                    X = pickle.load(pickle_in)

                    pickle_in = open("y.pickle","rb")
                    y = pickle.load(pickle_in)
                except OSError as e:
                    print("Processing training data..")
                    training_data = process_data("train", balance_to_lowest=True)

                    for features, labels in training_data:
                        X.append(features)
                        y.append(labels)

                    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                    y = np.array(y)

                    print("Saving training data..")
                    pickle_out = open("X.pickle","wb")
                    pickle.dump(X, pickle_out)
                    pickle_out.close()

                    pickle_out = open("y.pickle","wb")
                    pickle.dump(y, pickle_out)
                    pickle_out.close()

                #normalize to 0-1
                X = X/255.0

                #build a sequential convolutional NN, iterating over hyperparamters
                model = Sequential()

                model.add(Conv2D(conv_size, (filter_size, filter_size), input_shape=X.shape[1:], activation='relu'))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(conv_size, (filter_size, filter_size), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(conv_size, (filter_size, filter_size), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

                for i in range(dense_layers):
                    model.add(Dense(dense_size))
                    model.add(BatchNormalization())
                    
                model.add(Dropout(0.2))

                model.add(Dense(len(CATEGORIES)))
                model.add(Activation('sigmoid'))

                model.compile(loss='categorical_crossentropy',
                            optimizer='Adam',
                            metrics=['accuracy'])

                # trainable_count = int(
                #     np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
                # non_trainable_count = int(
                #     np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

                # print('Total params: {:,}'.format(trainable_count + non_trainable_count))
                # print('Trainable params: {:,}'.format(trainable_count))
                # print('Non-trainable params: {:,}'.format(non_trainable_count))
                X_val = []
                y_val = []

                for features, labels in process_data("val"):
                    X_val.append(features)
                    y_val.append(labels)

                X_val = np.array(X_val).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                X_val = X_val/255.0
                y_val = np.array(y_val)

                #init TensorBoard
                tensorBoard = TensorBoard(log_dir=f'logs/{model_name}')

                #creat checkpoint that saves at max val_accuracy
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filename,
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True)

                model.fit(X, y, batch_size=32, epochs=EPOCHS, validation_data=[X_val, y_val], callbacks=[tensorBoard, checkpoint])



