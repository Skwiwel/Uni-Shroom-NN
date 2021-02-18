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
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''\
            --------------------------------------------------
            Predict whether the mushroom is edible or 
            poisonous based on prevoiusly trained neural
            network.

            Part of a Warsaw University of Technology
            Faculty of Electronics and Information Technology
            project.
            --------------------------------------------------
            '''),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--inputFilename', required=True, type=pathlib.Path,
        help=textwrap.dedent('''\
            Input filename or path of the mushroom data to be classified.
            '''))
    parser.add_argument('-m', '--inputModel', default=defaultModelDataPath, type=pathlib.Path,
        help=textwrap.dedent('''\
            Input filename or path of the trained nn model.
            By default takes the model from shroom_nn/resources/
            '''))
    args = parser.parse_args()
    return args

def printResults(predictions, detailed):
    print(' {n1:>5}  {n2:<10}  {n3:<5}  {n4:>8}'.format(n1="row", n2="prediction", n3="aprox", n4="detailed"))
    for i in range(predictions.shape[1]):
        print(' {n1:>5}  {n2:<10}  {n3:<5}  {n4:>8}'.format(
                n1=i+1,
                n2=ediblePois(predictions[0,i]),
                n3="{:.0f}".format(predictions[0,i]),
                n4="{:.3f}".format(detailed[0,i])))

def ediblePois(x) -> str:
    if x == 1.0:
        return "poisonous"
    else:
        return "edible"

if __name__ == '__main__':
    args = parseArguments()

    _, data_x = cleanMushroomData(args.inputFilename)
    
    nn = NeuralNetwork()
    nn.loadFromFile(args.inputModel)

    predictions, detailed = nn.predict(data_x)
    printResults(predictions, detailed)