""" Main processing method to call. Takes folder path to source images as input, in addition to optional processing parameters.
    By default, the only processing performed is resizing to default values of 50x50. Alternate dimensions, greyscale conversion,
    and filtering with either 45º, 90º, or Gabor filters are also possible.
    Returns: A numpy array containing all images in the source directory, each formatted as numpy arrays, as well as a list of known 
    targets extracted from filenames.
    Note: Filenames should be named prefixed with 'cl' for cloudy, 'ra' for rainy, 'sh' for sunshine, and 'su' for sunrise."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Process(path : str, resize : bool = True, dim : tuple = (50,50), greyScale : bool = False, kernel : str = None) -> np.ndarray:

    tempList = []
    targetList = []
    classifications = {'cl' : 0, 'ra' : 1, 'sh' : 2, 'su' : 3}

    for filename in os.listdir(path):
        
        # f = open("imageshapes.txt", "a")
        
        # Ignore weird macOS hidden files
        if filename == '.DS_Store':
            continue

        print(os.path.join(path,filename))

        img = cv2.imread(os.path.join(path,filename))

        if img is None:
            continue

        img = img[:,:,::-1] # Reverse channels to match source image
        
        # apply pre-processing
        if resize == True:
            img, check = Resize(img, dim, filename)
            if check == 1:
                continue
        
        if greyScale == True:
            img = GreyScale(img)
        
        if kernel != None:
            img = Filter(img, kernel)

        tempList.append(img)
        targetList.append(classifications[filename[0:2]])
        print("Image Shape: " + str(img.shape))
        # f.write("Shape: " + str(img.shape) + "\n")
    
    outList = np.asarray(tempList)/255.0
    outTarget = np.asarray(targetList)

    print(outList.shape)
    return outList, outTarget


""" Helper processing functions that needn't be directly accessed."""

def Resize(img, dim : tuple, fileName : str):

    try:
        out = cv2.resize(img, dim)
        return out, 0
    
    except Exception as e:
        print("Failed to resize image: " + fileName)
        return None, 1


def GreyScale(img) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def Filter(img, kernel : str) -> np.ndarray:
    kernels = {'45deg' : np.array([[-1,-1,2], [-1,2,-1], [2,-1,-1]]),
               '90deg' : np.array([[0,-1,0], [-1,4,-1], [0,-1,0]]),
               'gabor' : cv2.getGaborKernel((21,21), 3.0, 0.0, 10.0, 0.5, 0, ktype = cv2.CV_32F)}
    
    return cv2.filter2D(img, -1, kernels[kernel])


# if __name__ == "__main__":
#     # For Testing:
#     test, target = Process("/Users/patrick/Google Drive/ENEL 525 - Machine Learning for Engineers/Course Project/Final Project/Tests", (500,500))
#     print(target)
    
#     for img in test:
#         plt.imshow(img)
#         plt.show()