# Import necessary libraries
import torch
import torch.nn as nn
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

# Ensure that 'hardest_examples.txt' exists in your directory
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

# Function to calculate the size of a model in bytes
def calculate_model_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size

# Function to simulate a training step
def train_step(model, X_batch, Y_batch, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(X_batch)
    loss = criterion(logits, Y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

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

# Function to restore kernels (attention heads) with zero bits (simulated here)
def restore_zero_bit_kernels(model, restore_fraction=0.5):
    restored_model = copy.deepcopy(model)
    # Simulate restoration by converting a fraction of weights back to full precision
    restored_params = list(restored_model.parameters())
    total_params = len(restored_params)
    num_restore = int(total_params * restore_fraction)
    for param in restored_params[:num_restore]:
        param.data = param.data.float()  # Restore precision
    return restored_model

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

# Define the loop variables
prev_safety_acc = 1.0  # Initialize with a high value (100% accuracy)
test_accs = []  # Store test accuracies over iterations
bytes_used = []  # Store model size in bytes after quantization
safety_losses = []  # Store safety set loss over iterations

# Parameters for the loop
safety_acc_drop_threshold = 0.05  # Threshold for acceptable accuracy drop (e.g., 5%)
restore_fraction = 0.5  # Fraction of kernels (attention heads, layers) to restore
weight_count = sum(p.numel() for p in model.parameters())  # Total number of weights in the model
max_iterations = 4000  # Number of training iterations

# Simulated quantization bits (could be dynamically adjusted)
Q = 8  # Number of bits for quantization (e.g., 8-bit quantization)

# Training loop with quantization and safety set evaluation
for i in range(max_iterations):
    # Step 1: Perform a training step and calculate loss
    idx = torch.randint(0, X_train.shape[0], (64,))  # Random batch
    X_batch, Y_batch = X_train[idx], Y_train[idx]
    loss = train_step(model, X_batch, Y_batch, optimizer, criterion)
    safety_loss = loss  # Using training loss as proxy for safety loss

    # Step 2: Calculate model size in bytes based on quantized bits
    model_bytes = Q / 8 * weight_count  # Q is the number of quantization bits

    # Step 3: Every 10 iterations:
    if i % 10 == 9:
        # Step 3.1: Calculate test accuracy
        test_acc = evaluate_accuracy(model, X_test, Y_test)

        # Step 3.2: Calculate accuracy on the preservation (safety) set
        safety_acc = evaluate_accuracy(model, X_pres, Y_pres)

        # Step 3.3: Compute accuracy drop from the previous safety evaluation
        acc_drop = prev_safety_acc - safety_acc

        # Step 3.4: If the drop exceeds the threshold
        if acc_drop > safety_acc_drop_threshold:
            # Step 3.5: Restore kernels with zero bits (e.g., restore 50% of them)
            model = restore_zero_bit_kernels(model, restore_fraction=restore_fraction)
            print(f"Iteration {i+1}: Safety accuracy drop detected. Restoring {restore_fraction*100}% of kernels.")

        # Step 3.7: Update the previous safety accuracy to the current one
        prev_safety_acc = safety_acc

        # Step 4: Log test accuracy, model size, and safety loss
        test_accs.append(test_acc)
        bytes_used.append(model_bytes)
        safety_losses.append(safety_loss)

        # Step 5: Print logging information
        if i % 100 == 99:
            print(f"Iteration {i+1}, Test Accuracy: {test_acc:.4f}, Safety Accuracy: {safety_acc:.4f}, Model Size: {model_bytes / (1024 * 1024):.4f} MB, Safety Loss: {safety_loss:.4f}")

    else:
        # Use the previous test accuracy
        if test_accs:
            test_acc = test_accs[-1]
        else:
            test_acc = 0.0

# After training, evaluate the final model on the test set
final_test_acc = evaluate_accuracy(model, X_test, Y_test)
print(f"Final Test Accuracy: {final_test_acc:.4f}")

# Calculate the final model size
final_model_size = calculate_model_size(model)
print(f"Final Model Size: {final_model_size / (1024 * 1024):.4f} MB")
