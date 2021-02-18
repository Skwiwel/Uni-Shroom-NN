import numpy as np

def splitTrainTest(x, y, ratio=0.8, seed=0):
    np.random.seed(seed)
    shuffled_x = np.array(x, copy=True)
    np.random.shuffle(shuffled_x.T)

    np.random.seed(seed)
    shuffled_y = np.array(y, copy=True)
    np.random.shuffle(shuffled_y.T)

    splitPoint = int(x.shape[1]*ratio)

    train_x = shuffled_x[:,:splitPoint]
    train_y = shuffled_y[:,:splitPoint]
    
    test_x = shuffled_x[:,splitPoint:]
    test_y = shuffled_y[:,splitPoint:]

    return train_x, train_y, test_x, test_y