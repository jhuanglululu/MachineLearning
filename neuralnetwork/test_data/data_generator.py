import numpy as np

from neuralnetwork.test_data import square_question_answer, square_root_question_answer, sum_question_answer

class DataGenerator:
    PAD_TOKEN = '<PAD>'
    EOS_TOKEN = '<EOS>'  # End Of Sequence token

    def __init__(self):
        self.token_to_id = { }
        self.id_to_token = { }
        self.vocab_size = 0
        self.samples = []
        self._create_vocabulary()
        self._generate_samples()

    def _create_vocabulary(self):
        used_numbers = set()
        
        # Numbers from square questions (1-10)
        for i in range(1, 11):
            used_numbers.add(str(i))
        
        # Results from square questions (1, 4, 9, 16, 25, 36, 49, 64, 81, 100)
        for i in range(1, 11):
            used_numbers.add(str(i ** 2))
        
        # Perfect squares for square root questions (1, 4, 9, 16, 25, 36, 49, 64, 81, 100)
        perfect_squares = [i * i for i in range(1, 11)]
        for num in perfect_squares:
            used_numbers.add(str(num))
        
        # Results from sum questions (2-20)
        for i in range(1, 11):
            for j in range(1, 11):
                used_numbers.add(str(i + j))
        
        numbers = sorted(list(used_numbers), key=int)

        words = ['what', 'is', 'the', 'square', 'root', 'of', 'sum', 'and']

        all_tokens = [self.PAD_TOKEN] + words + numbers + [self.EOS_TOKEN]

        self.token_to_id = { token: i for i, token in enumerate(all_tokens) }
        self.id_to_token = { i: token for i, token in enumerate(all_tokens) }
        self.vocab_size = len(all_tokens)

        if self.token_to_id[self.PAD_TOKEN] != 0:
            pad_id = self.token_to_id[self.PAD_TOKEN]
            current_token_at_0 = self.id_to_token[0]

            self.token_to_id[self.PAD_TOKEN] = 0
            self.id_to_token[0] = self.PAD_TOKEN

            self.token_to_id[current_token_at_0] = pad_id
            self.id_to_token[pad_id] = current_token_at_0

    def _generate_samples(self):
        samples = []

        perfect_squares = [i * i for i in range(1, 11)]  # Squares from 4 to 100
        for num in perfect_squares:
            samples.append(square_root_question_answer(num))

        for num in range(1, 11):
            samples.append(square_question_answer(num))

        for i in range(1, 11):
            for j in range(1, 11):
                samples.append(sum_question_answer(i, j))

        self.samples = samples

    def tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.token_to_id[self.PAD_TOKEN]) for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token[id] for id in ids]

    def create_batches(self, batch_size, shuffle=True):
        samples = self.samples.copy()

        if not samples:
            return []

        if shuffle:
            np.random.shuffle(samples)

        id_sequences = [self.tokens_to_ids(sample) for sample in samples]

        padded_sequences = self._pad_sequences(id_sequences)

        batches = []
        for i in range(0, len(padded_sequences), batch_size):
            batch_sequences = padded_sequences[i:i + batch_size]

            batch_array = np.array(batch_sequences)

            x = batch_array[:, :-1]
            y = batch_array[:, 1:]

            batches.append((x, y))

        return batches

    def _pad_sequences(self, sequences):
        if not sequences:
            return []

        max_length = max(len(seq) for seq in sequences)
        pad_id = self.token_to_id[self.PAD_TOKEN]
        padded = []

        for seq in sequences:
            padded_seq = seq + [pad_id] * (max_length - len(seq))
            padded.append(padded_seq)

        return padded

    def inspect(self):
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Total Samples: {len(self.samples)}")
        print()
        
        print("Vocabulary:")
        for i, token in enumerate(self.id_to_token.values()):
            print(f"  {i:2d}: '{token}'")
        print()
        
        print("Sample Data (first 5 examples):")
        for i, sample in enumerate(self.samples[:5]):
            tokens_str = ' '.join(sample)
            ids = self.tokens_to_ids(sample)
            print(f"  {i+1}. {tokens_str}")
            print(f"     IDs: {ids}")
        print()
        
        square_root_count = sum(1 for s in self.samples if 'root' in s)
        square_count = sum(1 for s in self.samples if 'square' in s and 'root' not in s)
        sum_count = sum(1 for s in self.samples if 'sum' in s)
        
        print("Sample Distribution:")
        print(f"  Square root questions: {square_root_count}")
        print(f"  Square questions: {square_count}")
        print(f"  Sum questions: {sum_count}")
        print()
        
        lengths = [len(sample) for sample in self.samples]
        print("Sequence Length Statistics:")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Average length: {sum(lengths) / len(lengths):.1f}")
        print()
        
        batch_example = self.create_batches(batch_size=2, shuffle=False)
        if batch_example:
            x_batch, y_batch = batch_example[0]
            print("Batch Example (first batch with batch_size=2):")
            print(f"  Input batch shape: {x_batch.shape}")
            print(f"  Target batch shape: {y_batch.shape}")
            print(f"  Input sequence 1: {x_batch[0]}")
            print(f"  Target sequence 1: {y_batch[0]}")
            print(f"  Input tokens 1: {[self.id_to_token[id] for id in x_batch[0]]}")
            print(f"  Target tokens 1: {[self.id_to_token[id] for id in y_batch[0]]}")

if __name__ == '__main__':
    data_gen = DataGenerator()
    data_gen.inspect()