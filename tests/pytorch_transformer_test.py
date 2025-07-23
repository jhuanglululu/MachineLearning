import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuralnetwork.test_data.data_generator import DataGenerator

# Same hyperparameters as the original transformer
d_model = 32
num_heads = 2
d_ff = 4 * d_model
n_layer = 1

n_batch = 100
batch_size = 16
learning_rate = 0.001
epoch_p_batch = 5

device = torch.device('cpu')
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layer, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.embedding.weight.data.normal_(0, math.sqrt(1.0 / d_model))
        self.output_projection.weight.data.normal_(0, math.sqrt(2.0 / d_model))

    def create_mask(self, src, src_pad_idx=0):
        batch_size, seq_len = src.shape
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Create padding mask
        padding_mask = (src == src_pad_idx)
        
        return causal_mask.to(src.device), padding_mask

    def forward(self, src):
        batch_size, seq_len = src.shape
        
        # Create masks
        causal_mask, padding_mask = self.create_mask(src)
        
        # Embedding and positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        
        # Transformer
        output = self.transformer(
            src_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # Output projection (no softmax)
        logits = self.output_projection(output)
        return logits

def convert_numpy_batch_to_torch(x_batch, y_batch):
    x_tensor = torch.from_numpy(x_batch).long().to(device)
    y_tensor = torch.from_numpy(y_batch).long().to(device)
    return x_tensor, y_tensor

def train_model():
    # Initialize data generator
    data_gen = DataGenerator()
    
    # Initialize model
    model = TransformerModel(
        vocab_size=data_gen.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        n_layer=n_layer
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training on {device}")
    print("\nStarting training...")
    
    # Training loop
    for batch_idx in range(n_batch):
        batches = data_gen.create_batches(batch_size, shuffle=True)
        
        batch_loss = 0.0
        for epoch in range(epoch_p_batch):
            epoch_loss = 0.0
            
            for x_batch, y_batch in batches:
                x_tensor, y_tensor = convert_numpy_batch_to_torch(x_batch, y_batch)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(x_tensor)
                
                # Calculate loss
                loss = criterion(logits.reshape(-1, data_gen.vocab_size), y_tensor.reshape(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            batch_loss += epoch_loss
        
        avg_loss = batch_loss / (epoch_p_batch * len(batches))
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Batch {batch_idx + 1}/{n_batch}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return model, data_gen

def generate_response(model, data_gen, question_text, max_length=20):
    model.eval()
    
    # Tokenize input
    input_tokens = question_text.split()
    try:
        input_ids = data_gen.tokens_to_ids(input_tokens)
    except KeyError as e:
        print(f"Unknown token: {e}")
        return None
    
    current_sequence = input_ids.copy()
    response_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Convert to tensor
            input_tensor = torch.tensor(current_sequence).unsqueeze(0).long().to(device)
            
            # Get predictions
            logits = model(input_tensor)
            
            # Get last token prediction and convert to probabilities
            last_token_logits = logits[0, len(current_sequence) - 1, :]
            last_token_probs = F.softmax(last_token_logits, dim=-1)
            
            # Sample from probability distribution
            predicted_id = torch.multinomial(last_token_probs, 1).item()
            
            predicted_token = data_gen.id_to_token[predicted_id]
            response_tokens.append(predicted_token)
            current_sequence.append(predicted_id)
            
            if predicted_token == data_gen.EOS_TOKEN:
                break
    
    return ' '.join(response_tokens)

def interactive_test(model, data_gen):
    print("\n" + "="*50)
    print("Interactive Testing (type 'exit' to quit)")
    print("="*50)
    
    while True:
        user_input = input("\nEnter a question: ")
        
        if user_input.lower() == 'exit':
            break
        
        response = generate_response(model, data_gen, user_input)
        if response:
            print(f"Response: {response}")
        else:
            print("Could not generate response due to unknown tokens.")
            print(f"Available tokens: {list(data_gen.token_to_id.keys())}")

if __name__ == "__main__":
    model, data_gen = train_model()
    interactive_test(model, data_gen)