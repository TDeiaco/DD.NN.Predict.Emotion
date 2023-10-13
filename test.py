import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.utils import class_weight

IMG_SIZE = 48

#CATEGORIES = ["angry", "fear", "happy", "sad", "surprise", "neutral"]
CATEGORIES = [ "happy", "sad", "neutral"]
SETS = ["test", "train", "val"]
NAME = f"EmoteCNN-1697070787"

targets = np.array([list(range(len(CATEGORIES)))]).reshape(-1)
one_hot_targets = np.eye(len(CATEGORIES))[targets]

#creates an array of X, y values, where X is the input img vector, and X are the classifications/labels
def process_data(set):
    data = []
    for category in CATEGORIES:  # do each category
        path = os.path.join(f"data/{set}",f"{category}")  # create path to each category
        for img in tqdm(os.listdir(path)):  # iterate over each image per each category
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

                data.append([new_array, one_hot_targets[CATEGORIES.index(category)]])  # add this to our training_data
                #data.append([new_array, CATEGORIES.index(category)])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

    random.shuffle(data)
    return data

model = tf.keras.models.load_model(f"models/{NAME}")

test_data = process_data('test')

X_test = []
y_test = []
for features, labels in test_data:
        X_test.append(features)
        y_test.append(labels)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

predictions = model.predict(X_test)
print(predictions)

pred_index = 0
for prediction in predictions[100:25]:
    print(f"Prediction: {CATEGORIES[np.argmax(prediction)]} Actual: {CATEGORIES[np.argmax(y_test[pred_index])]}")
    plt.imshow(X_test[pred_index])
    pred_index += 1
    plt.show()

