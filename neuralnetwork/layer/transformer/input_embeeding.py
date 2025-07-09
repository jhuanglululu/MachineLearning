from neuralnetwork.layer import Dense, SoftMax
from neuralnetwork.lossfunction import mse, mse_prime
from neuralnetwork.network import Network

class InputEmbedding:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        word_count = vocabulary.word_count()
        vector_dimension = 16
        embedding_layer = Dense(word_count, vector_dimension)

        network = Network([
            embedding_layer,
            Dense(vector_dimension, word_count),
            SoftMax()], (mse, mse_prime))

        input_word, output_word = vocabulary.get_test_data()
        network.train(input_word, output_word, 100)

        self.embedding = embedding_layer.weights.transpose()

    def get_embeddings(self, words):
        return [self.embedding[self.vocabulary.to_1_of_n(word)] for word in words]
