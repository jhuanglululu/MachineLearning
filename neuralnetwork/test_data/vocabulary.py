# Sentence Structure in QA format
# Ex1. what is the square of 3
# Ex2. what is the sum of 2 and 5
# Ex3. the square of 3 is 9
# Ex4. the square of 2 and 5 is 7

from neuralnetwork.test_data import *
import numpy as np

def _sliding_pairs(lst):
    return list(zip(lst, lst[1:]))

class Vocabulary:

    def __init__(self):
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                   '25', '36', '49', '64', '81', '100']

        words = ['what', 'is', 'the', 'square', 'root', 'of', 'sum', 'and', 'null']

        self.vocabs = numbers + words

    def word_count(self) -> int:
        return len(self.vocabs)

    def to_1_of_n(self, word: str) -> int:
        return self.vocabs.index(word)

    def to_word(self, n: int) -> str:
        return self.vocabs[n]

    def to_vector(self, word):
        vector = np.zeros((self.word_count(), 1))
        vector[self.to_1_of_n(word)] = 1
        return vector

    def from_vector(self, vector):
        index = np.argmax(vector)
        return self.to_word(index)

    def get_test_data(self):
        result = []
        # square
        for i in range(1, 11):
            for data in _sliding_pairs(square_question_answer(i)):
                if list(data) not in result:
                    result.append(list(data))

            for data in _sliding_pairs(square_root_question_answer(i)):
                if list(data) not in result:
                    result.append(list(data))

            for j in range(1, 11):
                for data in _sliding_pairs(sum_question_answer(i, j)):
                    if list(data) not in result:
                        result.append(list(data))

        input_word = [self.to_vector(word[0]) for word in result]
        output_word = [self.to_vector(word[1]) for word in result]

        return input_word, output_word
