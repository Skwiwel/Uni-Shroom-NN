#!/usr/bin/env python3.9

import sys
import io 

import numpy as np

from shroom_nn.utility.errprint import errprint


def cleanMushroomDataFromStr(dataString: str):
    return cleanMushroomData(io.StringIO(dataString))

def cleanMushroomData(datafile):
    data = np.genfromtxt(datafile, dtype=str, delimiter=',', filling_values=None)
    if data.ndim == 1:
        data = np.reshape(data, (1, -1))
    # if there are only 22 values in each set assume that y is missing.
    xStartIndex = 1
    if data.shape[1] == 22:
        xStartIndex = 0

    data = mushroomEncode(data)

    data_x = np.array(data[:,xStartIndex:], ndmin=2).T
    if xStartIndex == 1:
        data_y = np.array(data[:,0], ndmin=2)
    else:
        data_y = None
    return data_y, data_x

def mushroomEncode(data):
    rows = []
    for row in data:
        newRow = []
        for i, v in np.ndenumerate(row):
            index = i[0]
            # If the y data is missing: skip y
            if row.size == 22:
                index += 1
            definedValuesList = {
                0: ['p', 'e'],
                1: ['b', 'c', 'x', 'f', 'k', 's'],
                2: ['f', 'g', 'y', 's'],
                3: ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
                4: ['t', 'f'],
                5: ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
                6: ['a', 'd', 'f', 'n'],
                7: ['c', 'w', 'd'],
                8: ['b', 'n'],
                9: ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
                10: ['e', 't'],
                11: None, # ['b', 'c', 'u', 'e', 'z', 'r', '?'], Possibly fix missing values in the future
                12: ['f', 'y', 'k', 's'],
                13: ['f', 'y', 'k', 's'],
                14: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
                15: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
                16: ['p', 'u'],
                17: ['n', 'o', 'w', 'y'],
                18: ['n', 'o', 't'],
                19: ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
                20: ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
                21: ['a', 'c', 'n', 's', 'v', 'y'], # Possibly change encoding to numerical in this case
                22: ['g', 'l', 'm', 'p', 'u', 'w', 'd']
            }[index]
            if definedValuesList is None:
                continue
            newRow += oneHotEncode(v, definedValuesList)
        rows.append(newRow)
    return np.array(rows)
    
            

def oneHotEncode(v, vList: []) -> []:
    try:
        i = vList.index(v)
    except ValueError:
        errprint("Error: Could not encode value "+v+" to the set "+vList)
        sys.exit(1)

    # If just two values are present encode as 0 or 1
    if len(vList) == 2:
        if i == 0:
            return [1.0]
        else:
            return [0.0]

    arr = [0.]*len(vList)
    arr[i] = 1.0
    return arr

    
