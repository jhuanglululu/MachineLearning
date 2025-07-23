import time

import numpy as np

class BatchNetwork:

    def __init__(self, layers, loss_functions, data_generator=None):
        self.layers = layers
        self.loss_function, self.loss_function_prime = loss_functions
        self.data_generator = data_generator

    def forward(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train_batch(self, input_batch, target_batch, learning_rate=0.1):
        output = self.forward(input_batch)
        target_one_hot = self._create_one_hot(target_batch)

        loss = self.loss_function(output, target_one_hot)
        gradient = self.loss_function_prime(output, target_one_hot)

        for i, layer in enumerate(reversed(self.layers)):
            gradient = layer.backward(gradient, learning_rate)

        return loss

    def train(self, batch, iterations, learning_rate=0.1, show_error=None):
        for iteration in range(iterations):
            t1 = time.time()
            total_loss = 0
            num_batches = len(batch)

            for input_batch, target_batch in batch:
                loss = self.train_batch(input_batch, target_batch, learning_rate)
                total_loss += loss

            avg_loss = total_loss / num_batches

            if show_error is not None:
                print(
                    f"Epoch: {show_error['epoch'] + iteration} | Batch Number / Iteration: {show_error['batch_number']} - {iteration + 1} | Loss: {avg_loss:.6f} | Time: {time.time() - t1: .6f}")

    def _create_one_hot(self, target_batch):
        if self.data_generator is None:
            raise ValueError("Data generator is required for one-hot encoding")

        batch_size, seq_len = target_batch.shape
        vocab_size = self.data_generator.vocab_size

        one_hot = np.eye(vocab_size)[target_batch.flatten()]
        one_hot = one_hot.reshape(batch_size, seq_len, vocab_size)

        return one_hot

    def evaluate(self, data):
        """Evaluate the network on input data, returning softmax probabilities."""
        return self.forward(data)
