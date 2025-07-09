import numpy as np

from neuralnetwork.layer import *
from neuralnetwork.network.network import Network
from neuralnetwork.lossfunction import mse, mse_prime

def test_single_prediction(network, test_input):
    test_data = np.reshape(test_input, (len(test_input), 1))
    prediction = network.evaluate(test_data)
    a, b, c = test_input
    expected = [1 if a > b else 0, 1 if b > c else 0]

    print(f"\n--- Single Prediction Test ---")
    print(f"Input: [a={a}, b={b}, c={c}]")
    print(f"Expected: {expected} (a>{b}={expected[0]}, b>{c}={expected[1]})")
    print(f"Predicted: [{prediction[0]:.3f}, {prediction[1]:.3f}]")
    print(f"Rounded: [{round(prediction[0])}, {round(prediction[1])}]")
    print(f"Correct: {expected == [round(prediction[0]), round(prediction[1])]}\n")

network = Network([Dense(3, 5), Sigmoid(), Dense(5, 2), Sigmoid()], (mse, mse_prime))

# Rule:
#   input = (a, b, c)
#   output = (a > b, b > c)

data = [[2, 3, 4], [1, 5, 2], [3, 2, 3], [4, 1, 5], [0, 7, 3],
        [5, 2, 1], [2, 8, 4], [6, 1, 2], [3, 4, 6], [1, 9, 3]]

result = [[0, 0], [0, 1], [1, 0], [1, 0], [0, 1],
          [1, 1], [0, 1], [1, 0], [0, 0], [0, 1]]

X = np.reshape(data, (len(data), len(data[0]), 1))
Y = np.reshape(result, (len(result), len(result[0]), 1))

network.train(X, Y, epochs=1000, learning_rate=0.1)

test_case = input()
while test_case != "":
    test_single_prediction(network, [int(x) for x in test_case.split(" ")])
    test_case = input()
