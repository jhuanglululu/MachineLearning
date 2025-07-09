class Network:

    def __init__(self, layers, loss_functions):
        self.layers = layers
        self.loss_function, self.loss_function_prime = loss_functions

    def _predict(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, data, result, epochs, learning_rate=0.1, show_error=False):
        for e in range(epochs):
            error = 0
            for x, y in zip(data, result):
                output = self._predict(x)

                error += self.loss_function(y, output)

                gradient = self.loss_function_prime(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
            error /= len(data)

            if show_error:
                print(f"Epoch {e}: Average Error = {error:.6f}")

    def evaluate(self, data):
        return self._predict(data).flatten()
