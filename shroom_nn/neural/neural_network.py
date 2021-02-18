import numpy as np
from scipy.special import expit

class NeuralNetwork:
    _weights1 = np.zeros((1,1))
    _bias1 = np.zeros((1,1))
    _weights2 = np.zeros((1,1))
    _bias2 = np.zeros((1,1))

    _inputSize = 0
    _layer1Size = 0
    _layer2Size = 0

    _learningRate = 0.01

    def __init__(self, layer1Size=100, learningRate=0.01):
        self._layer1Size = layer1Size
        self._learningRate = learningRate

    def _sigmoid(self, x: float) -> float:
        return expit(x) # optimized implementation of 1/(1+np.exp(-x))

    def _sigmoidDerivative(self, x: float) -> float:
        sig = self._sigmoid(x)
        return sig * (1-sig)

    def trainModel(self, x, y, epochCount: int, seed=None):
        self._loadTrainingData(x, y)
        self._randomizeWeights(seed)
        for _ in range (epochCount):
            _, cache = self._feedForward(self._x)
            self._propagateBackward(self._x, self._y, cache)

    def _loadTrainingData(self, x, y):
        self._inputSize = x.shape[0]
        self._layer2Size = y.shape[0]
        self._y = y
        self._x = x
        self._resetWeights()

    def _resetWeights(self):
        self._weights1 = np.zeros((self._layer1Size, self._inputSize))
        self._bias1 = np.zeros((self._layer1Size, 1))    # bias weight set to 0 
        self._weights2 = np.zeros((self._layer2Size, self._layer1Size))
        self._bias2 = np.zeros((self._layer2Size, 1))    # bias weight set to 0
        self._last_d_weights1 = np.zeros(self._weights1.shape)
        self._last_d_weights2 = np.zeros(self._weights2.shape)

    def _randomizeWeights(self, seed=None):
        np.random.seed(seed)
        self._weights1 = np.random.rand(self._weights1.shape[0], self._weights1.shape[1]) 
        self._bias1 = np.zeros((self._layer1Size, 1))    # bias weight set to 0 
        self._weights2 = np.random.rand(self._weights2.shape[0], self._weights2.shape[1]) 
        self._bias2 = np.zeros((self._layer2Size, 1))    # bias weight set to 0 

    def _getAvgCost(self, out, y) -> float:
        return np.average(np.absolute(y - out))

    def _feedForward(self, x):
        Z1 = np.dot(self._weights1, x) + self._bias1
        A1 = self._sigmoid(Z1)
        A2 = np.dot(self._weights2, A1) + self._bias2
        return A2, {"Z1": Z1, "A1": A1, "A2": A2}

    _epoch = 0

    def _propagateBackward(self, x, y, cache):
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        A2 = cache["A2"]

        d_weights2 = np.zeros(self._weights2.shape)
        d_weights1 = np.zeros(self._weights1.shape)

        inputCount = x.shape[1]
        
        d_error = 2*(A2 - y)

        d_weights2 = np.dot(d_error, A1.T) / inputCount        
        d_bias2 = np.sum(d_error, axis=1, keepdims=True) / inputCount 
        
        d_A1 = np.dot(self._weights2.T, d_error)
        d_Z1 = np.multiply(d_A1, self._sigmoidDerivative(Z1))
        d_weights1 = np.dot(d_Z1, x.T)
        d_bias1 = np.sum(d_Z1, axis=1, keepdims=True) / inputCount 

        d_weights2 = self._learningRate * d_weights2
        d_weights1 = self._learningRate * d_weights1
        d_bias2 = self._learningRate * d_bias2
        d_bias1 = self._learningRate * d_bias1
        self._weights2 -= d_weights2
        self._weights1 -= d_weights1
        self._bias2 -= d_bias2
        self._bias1 -= d_bias1

        self._last_d_weights2 = d_weights2
        self._last_d_weights1 = d_weights1
        self._last_d_bias2 = d_bias2
        self._last_d_bias1 = d_bias1

    def predict(self, x):
        result, _ = self._feedForward(x)
        roundedPredictions = np.vectorize(self._roundTo01)(result)
        return roundedPredictions, result

    def _roundTo01(self, x: float) -> int:
        if x >= 0.5:
            return 1
        else:
            return 0

    def saveToFile(self, path):
        np.savez_compressed(
            path, 
            inS = self._inputSize,
            l1S = self._layer1Size,
            l2S = self._layer2Size,
            w1 = self._weights1,
            b1 = self._bias1,
            w2 = self._weights2,
            b2 = self._bias2
        )

    def loadFromFile(self, path):
        data = np.load(path, allow_pickle=False)
        self._inputSize = data['inS']
        self._layer1Size = data['l1S']
        self._layer2Size = data['l2S']
        self._weights1 = data['w1']
        self._bias1 = data['b1']
        self._weights2 = data['w2']
        self._bias2 = data['b2']
