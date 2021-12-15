""" Helper module to extract a random selection of images from the training data set to be used for testing purposes."""

import numpy as np
import random

def ExtractRandomTest(trainingSet : np.ndarray, targetList : np.ndarray, percentTest : float = 0.10) -> np.ndarray:

    trainingSize = len(trainingSet)
    testImagesCount = int(percentTest * float(trainingSize))
    print(str(testImagesCount) + " images will be randomly selected as test images.\n")

    indexList = random.sample(range(0, trainingSize - 1), testImagesCount)

    testSet = trainingSet[indexList]
    testTarget = np.take(targetList, indexList)

    print("Shape of original training set: " + str(trainingSet.shape))
    print("Shape of output test set: " + str(testSet.shape))

    return testSet, testTarget