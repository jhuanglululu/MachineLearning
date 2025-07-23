import os
import sys

from neuralnetwork.layer import SoftMax
from neuralnetwork.layer.transformer import AddAndNorm, TransformerFFN
from neuralnetwork.layer.transformer.attention import MultiHeadAttention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from neuralnetwork.layer.debug import ShapeDebugLayer, StatDebugLayer
from neuralnetwork.layer.transformer.embedding_projection import create_shared_embedding_projection
from neuralnetwork.layer.transformer.positional_encoding import PositionalEncoding
from neuralnetwork.lossfunction import cross_entropy, cross_entropy_prime
from neuralnetwork.network.batch_network import BatchNetwork
from neuralnetwork.test_data.data_generator import DataGenerator

d_model = 64
num_heads = 4
d_ff = 4 * d_model
n_layer = 2

n_batch = 200
batch_size = 16
learning_rate = 0.00001
epoch_p_batch = 50

def transformer_layer(number):
    return [
        layer
        for _ in range(number)
        for layer in [
            AddAndNorm(d_model, MultiHeadAttention(d_model, num_heads)),
            AddAndNorm(d_model, TransformerFFN(d_model, d_ff))
        ]
    ]

data_gen = DataGenerator()
vocab_size = data_gen.vocab_size

embedding, projection = create_shared_embedding_projection(vocab_size, d_model)
positional_encoding = PositionalEncoding(d_model)

transformer = BatchNetwork(
    [embedding,
     positional_encoding,#StatDebugLayer('After Encoding'),
     *transformer_layer(n_layer),#StatDebugLayer('After Transformer'),
     projection,#StatDebugLayer('After Projection')
     ],
    (cross_entropy, cross_entropy_prime),
    data_generator=data_gen)

epoch = 1
for i in range(n_batch):
    batches = data_gen.create_batches(batch_size=batch_size)
    loss = transformer.train(batches, iterations=epoch_p_batch, learning_rate=learning_rate,
                             show_error={ 'epoch': epoch, 'batch_number': i + 1 })
    epoch += epoch_p_batch

# Interactive Loop
softmax = SoftMax()
while True:
    user_input = input("Enter a question (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break

    input_question_tokens = user_input.split()
    try:
        question_ids = data_gen.tokens_to_ids(input_question_tokens)
    except KeyError as e:
        print(f"Unknown token: {e}")
        print(f"Available tokens: {list(data_gen.token_to_id.keys())}")
        continue

    current_sequence = question_ids.copy()
    response_tokens = []

    max_response_length = 20

    for _ in range(max_response_length):
        question_batch = np.array(current_sequence).reshape(1, -1)

        prediction = transformer.forward(question_batch)

        probs = prediction[0, len(current_sequence) - 1, :]
        predicted_id = np.random.choice(len(probs), p=softmax.forward(probs))
        predicted_token = data_gen.id_to_token[predicted_id]

        response_tokens.append(predicted_token)
        current_sequence.append(predicted_id)

        if predicted_token == data_gen.EOS_TOKEN:
            break

    print(f"Response: {' '.join(response_tokens)}")
    print("---")
