# Weather_Classifier
 Tensorflow based convolutional neural network to classify weather in images.

Created by: Patrick Willmann for ENEL 525 - Introduction to Machine Learning at the Schulich School of Engineering

The main program (WeatherClassifier.py) takes two command line arguments: 
- A file path to the training dataset folder 
- A file path to a test dataset folder

If the training dataset is passed in the test dataset argument, 10% of the training images will be randomly selected for testing purposes. 

Images will be classified as follows: Cloudy, Rainy, Sunshine, and Sunrise.

Filenames for training and test data should be named prefixed with:
'cl' for cloudy, 'ra' for rainy, 'sh' for sunshine, and 'su' for sunrise

This is required for training and model evaluation purposes. 

No training data is included in this repository. However, the publicly available dataset I used can be found here: https://data.mendeley.com/datasets/4drtyfjtfy/
1

(Citation: Ajayi, Gbeminiyi (2018), “Multi-class Weather Dataset for Image Classification”, Mendeley Data, V1, doi: 10.17632/4drtyfjtfy.1)

Note: A mashed together .jpynb version of the code has been included to easily import and run in Google Colab. Training and test data paths will need to be added to the code manually here, rather than passed as command line arguments. 