""" Weather Classification Image Recognition using Google Tensorflow
    For ENEL 525 - Introduction to Machine Learning at the Schulich School of Engineering
    Created by: Patrick Robert Willmann

    Images will be classified as follows: Cloudy, Rainy, Sunshine, and Sunrise.
    Note: Filenames for training and test data should be named prefixed with:
    'cl' for cloudy, 'ra' for rainy, 'sh' for sunshine, and 'su' for sunrise.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import ProcessImages as pr, ExtractRandomTest as ex, Training as tr

if __name__ == "__main__":

    print("\nWeather Classification Image Recognition using Google Tensorflow\nFor ENEL 525 - Introduction to Machine Learning at the Schulich School of Engineering\nCreated by: Patrick Robert Willmann\n\nImages will be classified as follows: Cloudy, Rainy, Sunshine, and Sunrise.\n\nNote: Filenames for training and test data should be named prefixed with:\n'cl' for cloudy, 'ra' for rainy, 'sh' for sunshine, and 'su' for sunrise.")

    # Initialize Default Values
    customTest = False
    dataPaths = []
    dimensions = (100,100)
    epochs = 12

    # Parse Input Args

    if len(sys.argv) != 3:
        print("Weather Classifier\n\nExpected Arguments:\n1) Path to Training Data\n2) Path to Testing Data (can be the same as the training data - a randomized subset of 1/10 of the training images will be chosen if the same)\n")
        exit(1)

    else:
        
        print("Using Tensorflow Version: " + tf.version.VERSION + "\n")
        
        if os.path.exists(sys.argv[1]):
            dataPaths.append(str(sys.argv[1]))
            print("Training Data Path: " + dataPaths[0])
        else:
            print("Training data path provided does not exist!\nExiting...\n")
            exit(2)
        
        if os.path.exists(sys.argv[2]):
            dataPaths.append(str(sys.argv[2]))
            print("Test Data Path: " + dataPaths[1])
        else:
            print("Test data path provided does not exist!\n Exiting...\n")
            exit(2)

        if dataPaths[0] != dataPaths[1]:
            customTest = True


    # Import & Process Images

    print("\nProcessing training images...")
    trainingImages, trainingTargets = pr.Process(dataPaths[0], dim = dimensions)

    if customTest == True:
        print("Processing test images...")
        testImages, testTargets = pr.Process(dataPaths[1], dim = dimensions)
    else:
        # Take a subset of images from trainingImages list to testImages list
        print("Extracting test images from training dataset...")
        testImages, testTargets = ex.ExtractRandomTest(trainingImages, trainingTargets)


    # Train and Test the Pre-Processed Dataset

    testLoss, testAcc = tr.Train(trainingImages,trainingTargets, testImages, testTargets, dimensions, epochs)

    print("Test Accuracy: " + str(testAcc))
    print("Test Losses: " +str(testLoss) + "\n")
    
    testResults = tr.Test(testImages)
    testResults = testResults.astype(int)
    confusionMatrix = tr.compute_confusion_matrix(testTargets, testResults)
    
    print("\nTarget Values: " + str(testTargets))
    print("Test Results: " + str(testResults) + "\n")
    print("Confusion Matrix:\n" + str(confusionMatrix) + "\n")

    # Display images and their predictions and targets if 10 or fewer test images are provided
    if (len(testImages) <= 10):
        fig = plt.figure(figsize = (10, 7))
        rows = 2
        cols = 5
        classifications = {0: 'cl', 1: 'ra', 2: 'sh', 3: 'su'}
        for i, img in enumerate(testImages):
            fig.add_subplot(rows, cols, (i+1))
            plt.imshow(testImages[i])
            plt.title("P: " + str(classifications[testResults[i]]) + ", T: " + str(classifications[testTargets[i]]))
        plt.show()