import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from neuralnetwork.layer import Sigmoid, Dense
from neuralnetwork.layer.cnn import *
from neuralnetwork.lossfunction import bce, bce_prime
from neuralnetwork.network import Network

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indicies = np.hstack((zero_index, one_index))
    all_indicies = np.random.permutation(all_indicies)
    x, y = x[all_indicies], y[all_indicies]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

network = Network([
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
], (bce, bce_prime))

network.train(x_train, y_train, 20, show_error=True)

success = 0
fail = 0
for x, y in zip(x_test, y_test):
    output = network.evaluate(x)
    predicted = np.argmax(output)
    actual = np.argmax(y)
    print(f"predicted: {predicted}, actual: {actual}")
    if predicted == actual:
        success += 1
    else :
        fail += 1

total = success + fail
print(f"Sample: {total}")
print(f"Sucess: {success} ({100 * success / total:.3f}%)")
print(f"Fail: {fail} ({100 * fail / total:.3f}%)")