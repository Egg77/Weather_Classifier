""" This module contains the tensorflow convolutional neural network, including the neural network, fitting, and analysis functions."""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

model = models.Sequential()

def Train(trainingData : np.ndarray, trainingTargets : np.ndarray, testData : np.ndarray, testTargets : np.ndarray, dim : tuple, epochs : int = 10):

    print("Creating model...")
    model.add(layers.Conv2D(20, (3, 3), activation = 'relu', input_shape = (dim[0], dim[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(40, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(80, (3, 3), activation = 'relu'))

    # Flatten 3D layers into 1D then feed into Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(80, activation = 'relu'))
    model.add(layers.Dense(4))
    model.summary()

    # Compile, train, and test the model
    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

    history = model.fit(trainingData, trainingTargets, epochs = epochs, validation_data = (testData, testTargets))

    # Perform Analysis
    print("\n~~~~~~~~~Fitting Summary~~~~~~~~~~")
    print("\nFinal Epoch: Training Loss: " + str(history.history['loss'][epochs-1]) + "; Training Accuracy: " + str(history.history['accuracy'][epochs-1]))

    plt.plot(history.history['accuracy'], label = 'accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc = 'lower right')
    plt.show()

    return model.evaluate(testData, testTargets, verbose=2)


def Test(testData):
    return oneHotDec(model.predict(testData))


def compute_confusion_matrix(targetVals, predVals):
  K = len(np.unique(targetVals)) # Number of classes
  result = np.zeros((K, K))
  for i in range(len(targetVals)):
    result[targetVals[i]][predVals[i]] += 1
  return result


def oneHotDec(predVals):
  result = np.zeros(len(predVals))
  for i, val in enumerate(predVals):
    result[i] = (np.argmax(val))
  return result