# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load and preprocess the dataset
with open('names.txt', 'r') as f:
    words = f.read().splitlines()

# Create character to index mappings
chars = sorted(list(set(''.join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}  # Start indices from 1
stoi['.'] = 0  # End-of-sequence token
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)
print(f"Vocabulary size: {vocab_size}")

# Function to build dataset
def build_dataset(words, block_size):
    X, Y = [], []
    for word in words:
        context = [0] * block_size  # Initialize with start tokens
        for ch in word + '.':
            idx = stoi[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]  # Slide window
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y

# Split data into training, validation, and test sets
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
block_size = 8

X_train, Y_train = build_dataset(words[:n1], block_size)
X_val, Y_val = build_dataset(words[n1:n2], block_size)
X_test, Y_test = build_dataset(words[n2:], block_size)

print(f"Training set: {X_train.shape}, {Y_train.shape}")
print(f"Validation set: {X_val.shape}, {Y_val.shape}")
print(f"Test set: {X_test.shape}, {Y_test.shape}")

# Load preservation (safety) set
def load_preservation_set(file_path):
    with open(file_path, 'r') as f:
        preservation_words = [line.strip().lower() for line in f]
    return preservation_words

preservation_words = load_preservation_set('hardest_examples.txt')
X_pres, Y_pres = build_dataset(preservation_words, block_size)
print(f"Preservation set: {X_pres.shape}, {Y_pres.shape}")

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.embed_size = embed_size
        self.block_size = block_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.block_size, "Input sequence length exceeds block size"
        token_embeddings = self.token_embedding(x)  # (B, T, embed_size)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings  # (B, T, embed_size)
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.fc_out(x[:, -1, :])  # Predict next token
        return logits

# Initialize the model
embed_size = 128
num_heads = 8
num_layers = 4
dropout = 0.1

model = TransformerModel(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size,
    dropout=dropout
)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Function to evaluate accuracy
def evaluate_accuracy(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == Y).float().sum()
        accuracy = correct / Y.shape[0]
    return accuracy.item()

# Function to quantize the model
def quantize_model(model, bit_depth=8):
    model_int = copy.deepcopy(model)
    # Simulate quantization by reducing precision of weights
    for param in model_int.parameters():
        param.data = param.data.half()  # Convert to half precision (16-bit)
    return model_int

# Training loop with quantization and safety set evaluation
epochs = 5
batch_size = 64
loss_increase_threshold = 0.05  # 5% acceptable loss increase
best_accuracy = 0
patience = 2  # Number of epochs to wait before restoring precision
no_improve_epochs = 0

for epoch in range(epochs):
    # Evaluate on preservation set before training
    preservation_accuracy_before = evaluate_accuracy(model, X_pres, Y_pres)
    print(f"Epoch {epoch+1}, Preservation Accuracy Before Training: {preservation_accuracy_before:.4f}")

    model.train()
    # Shuffle training data
    perm = torch.randperm(X_train.size(0))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # Training batches
    for i in range(0, X_train.size(0), batch_size):
        X_batch = X_train[i:i+batch_size]
        Y_batch = Y_train[i:i+batch_size]

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

    # Update quantization parameters (simulated here)
    # For simplicity, we're not changing bit depth or scaling factor in this example

    # Quantize the model
    quantized_model = quantize_model(model)

    # Evaluate on preservation set after quantization
    preservation_accuracy_after = evaluate_accuracy(quantized_model, X_pres, Y_pres)
    print(f"Epoch {epoch+1}, Preservation Accuracy After Quantization: {preservation_accuracy_after:.4f}")

    # Check if accuracy drop exceeds threshold
    accuracy_drop = preservation_accuracy_before - preservation_accuracy_after
    if accuracy_drop > loss_increase_threshold:
        print("Accuracy drop exceeds threshold, restoring some attention heads to higher precision.")
        # Simulate restoring some attention heads
        # For simplicity, we'll restore the original model
        quantized_model = copy.deepcopy(model)
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("No improvement after multiple epochs, stopping training.")
            break
    else:
        # Update the model with quantized model
        model = quantized_model
        no_improve_epochs = 0

    # Evaluate on validation set
    val_accuracy = evaluate_accuracy(model, X_val, Y_val)
    print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model_state = copy.deepcopy(model.state_dict())

# Load best model and evaluate on test set
model.load_state_dict(best_model_state)
test_accuracy = evaluate_accuracy(model, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")