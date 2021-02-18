#!/usr/bin/env python3.9

import argparse
import textwrap
import sys
import pathlib

import numpy as np
import time
import random
 
from shroom_nn.neural.neural_network import NeuralNetwork
from shroom_nn.data.clean import cleanMushroomData
from shroom_nn.data.split import splitTrainTest

from shroom_nn.utility.errprint import errprint

def getPercentAccuracy(y, predicted_y) -> float:
    return round(getAccuracy(y, predicted_y) * 100, 2)

def getAccuracy(y, predicted_y) -> float:
    return float((np.dot(y, predicted_y.T) + np.dot(1. - y, 1. - predicted_y.T)) / float(y.size))

def parseArguments():
    defaultModelDataPath = "shroom_nn/resources/model.npz"
    defaultInputDataPath = "shroom_nn/resources/agaricus-lepiota.data"
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''\
            --------------------------------------------------
            Create and train a neural network model.
            Saves the resulting model in a file.

            Part of a Warsaw University of Technology
            Faculty of Electronics and Information Technology
            project.
            --------------------------------------------------
            '''),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--inputFilename', default=defaultInputDataPath, type=pathlib.Path,
        help=textwrap.dedent('''\
            Input filename or path of the data.
            By default takes the data from shroom_nn/resources/
            '''))
    parser.add_argument('-o', '--outputFilename', default=defaultModelDataPath, type=pathlib.Path,
        help=textwrap.dedent('''\
            Output filename for the resulting model data.
            By default saves to shroom_nn/resources/
            '''))
    parser.add_argument('-s', '--seed', default=None, type=int,
        help=textwrap.dedent('''\
            Seed for the initial model weights and data train-test splitting.
            By default uses a default random system dependent value.
            '''))
    parser.add_argument('-t', '--trainingFraction', default=0.8, type=float,
        help=textwrap.dedent('''\
            What fraction of the data is to be used as training data.
            The rest will be used as testing data.
            The fraction is a number in the range (0.0, 1.0]
            If a fraction of 1.0 is specified no accuracy data will be shown.
            Default: 0.5
            '''))
    parser.add_argument('-e', '--epochCount', default=100, type=int,
        help=textwrap.dedent('''\
            Number of epochs/iterations of the training.
            Default: 100
            '''))
    parser.add_argument('--hiddenLayerSize', default=10, type=int,
        help=textwrap.dedent('''\
            Number of the hidden layer (layer 1) neurons.
            Default: 10
            '''))
    parser.add_argument('-l', '--learningRate', default=0.01, type=float,
        help=textwrap.dedent('''\
            Learning rate of the algorithm.
            The learning rate is constant across the learning epochs.
            Default: 0.01
            '''))
    parser.add_argument('--shortOutput', action='store_true',
        help=textwrap.dedent('''\
            Print shortened output for batch testing.
            Will only print train set accuracy, test set accuracy and train time.
            '''))
    args = parser.parse_args()

    if (args.trainingFraction <= 0.0 or args.trainingFraction > 1.0):
        errprint("Error: The trainingFraction must be in the range (0.0, 1.0].")
        sys.exit(1)
    if (args.epochCount <= 0):
        errprint("Error: The epochCount must be bigger than 0.")
        sys.exit(1)
    if (args.hiddenLayerSize <= 0):
        errprint("Error: The hiddenLayerSize must be bigger than 0.")
        sys.exit(1)
    return args

if __name__ == '__main__':
    args = parseArguments()

    data_y, data_x = cleanMushroomData(args.inputFilename)
    if data_y is None:
        errprint("Error: Invalid mushroom dataset for training.")
        sys.exit(1)

    random.seed(args.seed)
    seedForSplitting = random.randrange(2**32 - 1)
    train_x, train_y, test_x, test_y = splitTrainTest(data_x, data_y, ratio=args.trainingFraction, seed=seedForSplitting)
    if train_x.size == 0:
        errprint("Error: Insufficient training data for the specified train/test split ratio.")
        sys.exit(1)

    if (args.shortOutput is not True):
        print("Training data:")
        print("train y data shape: "+str(train_y.shape))
        print("train x data shape: "+str(train_x.shape))
        print("test y data shape: "+str(test_y.shape))
        print("test x data shape: "+str(test_x.shape))
    
    nn = NeuralNetwork(args.hiddenLayerSize, args.learningRate)

    timeStart = time.perf_counter_ns()
    nn.trainModel(train_x, train_y, args.epochCount, seed=args.seed)
    timeEnd = time.perf_counter_ns()

    seconds = round((timeEnd - timeStart) / 1e+9, 2)

    predictions, _ = nn.predict(train_x)
    trainPercentAccuracy = getPercentAccuracy(train_y, predictions)
    testPercentAccuracy = None
    if (args.trainingFraction < 1.0):
        predictions, detailed = nn.predict(test_x)
        testPercentAccuracy = getPercentAccuracy(test_y, predictions)

    if (args.shortOutput is not True): 
        print("\nFinished training")
        print("time elapsed: "+str(seconds)+"s\n")
        print("Train data accuracy: " + str(trainPercentAccuracy) + "%")
        if (testPercentAccuracy is not None):
            print("Test data accuracy: " + str(testPercentAccuracy) + "%")
    else:
        print(str(trainPercentAccuracy) + "%" + " ", end='')
        if (testPercentAccuracy is not None):
            print(str(testPercentAccuracy) + "%" + " ", end = '')
        print(str(seconds)+"s")

    nn.saveToFile(args.outputFilename)
    if (args.shortOutput is not True): 
        print("\nModel data saved to:")
        print(args.outputFilename)