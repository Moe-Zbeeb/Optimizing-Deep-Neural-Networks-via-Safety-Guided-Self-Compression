# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

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

# Define the quantization function
def quantize(x, b, e):
    # x: input tensor
    # b: bit depth (must be >=1)
    # e: scaling factor
    x_scaled = x / (2 ** e)
    x_clipped = torch.clamp(x_scaled, -2 ** (b - 1), 2 ** (b - 1) - 1)
    x_quantized = torch.floor(x_clipped)
    x_dequantized = x_quantized * (2 ** e)
    return x_dequantized

# Define custom MultiheadAttention with quantization
class QuantizedMultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(QuantizedMultiheadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Initialize weight matrices for query, key, and value
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        # Initialize learnable parameters b and e for quantization
        self.b = nn.Parameter(torch.tensor(8.0))  # Start with 8 bits
        self.e = nn.Parameter(torch.tensor(0.0))  # Start with scaling factor 0

    def forward(self, values, keys, query):
        N, T, _ = query.size()

        # Linear projections
        queries = self.q_linear(query)
        keys = self.k_linear(keys)
        values = self.v_linear(values)

        # Split into multiple heads
        queries = queries.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply quantization to attention heads
        b = torch.clamp(self.b, min=1.0, max=8.0)
        e = self.e
        queries = quantize(queries, b, e)
        keys = quantize(keys, b, e)
        values = quantize(values, b, e)

        # Calculate attention scores
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embed_size ** (1/2))
        attention = torch.softmax(energy, dim=-1)

        # Get context
        out = torch.matmul(attention, values)
        out = out.transpose(1, 2).contiguous().view(N, T, self.embed_size)

        # Final linear layer
        out = self.fc_out(out)

        return out

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, block_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.embed_size = embed_size
        self.block_size = block_size
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': QuantizedMultiheadAttention(embed_size, num_heads),
                'ln1': nn.LayerNorm(embed_size),
                'ff': nn.Sequential(
                    nn.Linear(embed_size, embed_size * 4),
                    nn.GELU(),
                    nn.Linear(embed_size * 4, embed_size),
                    nn.Dropout(dropout)
                ),
                'ln2': nn.LayerNorm(embed_size)
            }) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.block_size, "Input sequence length exceeds block size"
        token_embeddings = self.token_embedding(x)  # (B, T, embed_size)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings  # (B, T, embed_size)

        for layer in self.layers:
            attn = layer['attn'](x, x, x)
            x = x + attn
            x = layer['ln1'](x)
            ff = layer['ff'](x)
            x = x + ff
            x = layer['ln2'](x)

        x = self.ln_f(x)
        logits = self.fc_out(x[:, -1, :])  # Predict next token
        return logits

    def get_quantization_params(self):
        b_list = []
        e_list = []
        for layer in self.layers:
            b_list.append(layer['attn'].b)
            e_list.append(layer['attn'].e)
        return b_list, e_list

# Function to calculate the size of a model in bytes
def calculate_model_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size

# Define custom loss function
def custom_loss_function(model, logits, targets, factor, preservation_loss):
    # Cross-entropy loss
    ce_loss = nn.CrossEntropyLoss()(logits, targets)

    # Regularization term proportional to model size
    b_list, _ = model.get_quantization_params()
    model_size = 0
    for b in b_list:
        model_size += b.mean()
    reg_loss = factor * model_size

    # Total loss
    total_loss = ce_loss + reg_loss + preservation_loss
    return total_loss

# Function to calculate preservation loss
def compute_preservation_loss(model, X_pres, Y_pres):
    model.eval()
    with torch.no_grad():
        logits = model(X_pres)
        loss = nn.CrossEntropyLoss()(logits, Y_pres)
    model.train()
    return loss

# Function to quantize the model
def quantize_model(model):
    for layer in model.layers:
        attn = layer['attn']
        b = torch.clamp(attn.b, min=1.0, max=8.0)
        e = attn.e
        # Quantize the weights
        with torch.no_grad():
            attn.q_linear.weight.data = quantize(attn.q_linear.weight.data, b, e)
            attn.k_linear.weight.data = quantize(attn.k_linear.weight.data, b, e)
            attn.v_linear.weight.data = quantize(attn.v_linear.weight.data, b, e)
            attn.fc_out.weight.data = quantize(attn.fc_out.weight.data, b, e)

# Function to restore kernels (attention heads) to half-precision
def restore_attention_heads(model, restore_fraction=0.5):
    num_layers = len(model.layers)
    num_restore = int(num_layers * restore_fraction)
    for layer in model.layers[:num_restore]:
        attn = layer['attn']
        # Restore b to 16 bits (simulate half-precision)
        with torch.no_grad():
            attn.b.data = torch.tensor(16.0)

# Function to simulate a training step
def train_step(model, X_batch, Y_batch, optimizer, factor, X_pres, Y_pres):
    optimizer.zero_grad()
    logits = model(X_batch)
    # Compute preservation loss
    preservation_loss = compute_preservation_loss(model, X_pres, Y_pres)
    # Compute total loss
    loss = custom_loss_function(model, logits, Y_batch, factor, preservation_loss)
    loss.backward()
    optimizer.step()
    return loss.item(), preservation_loss.item()

# Function to evaluate accuracy
def evaluate_accuracy(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == Y).float().sum()
        accuracy = correct / Y.shape[0]
    model.train()
    return accuracy.item()

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
factor = 1e-4  # Regularization factor

# Define the loop variables
prev_preservation_loss = float('inf')  # Initialize with a high value
test_accs = []  # Store test accuracies over iterations
model_sizes = []  # Store model size after quantization
preservation_losses = []  # Store preservation set loss over iterations

# Parameters for the loop
preservation_loss_threshold = 0.1   # Threshold for acceptable loss increase
restore_fraction = 0.5  # Fraction of attention heads to restore
max_iterations = 4000  # Number of training iterations

# Training loop with quantization and preservation set evaluation
for i in range(max_iterations):
    # Step 1: Perform a training step and calculate loss
    idx = torch.randint(0, X_train.shape[0], (64,))  # Random batch
    X_batch, Y_batch = X_train[idx], Y_train[idx]
    loss, preservation_loss = train_step(model, X_batch, Y_batch, optimizer, factor, X_pres, Y_pres)

    # Step 2: Quantize the attention heads after each weight update
    quantize_model(model)

    # Step 3: Periodically evaluate and adjust
    if i % 10 == 9:
        # Evaluate the model on the preservation set
        current_preservation_loss = compute_preservation_loss(model, X_pres, Y_pres)

        # If the preservation loss increases beyond a defined threshold after quantization
        loss_increase = current_preservation_loss - prev_preservation_loss
        if loss_increase > preservation_loss_threshold:
            # Restore some attention heads to their original half-precision
            restore_attention_heads(model, restore_fraction)
            print(f"Iteration {i+1}: Preservation loss increased by {loss_increase:.4f}. Restoring {restore_fraction*100}% of attention heads.")

        # Otherwise, keep the quantized model
        prev_preservation_loss = current_preservation_loss

        # Evaluate test accuracy
        test_acc = evaluate_accuracy(model, X_test, Y_test)

        # Calculate model size
        model_size = calculate_model_size(model)

        # Logging
        test_accs.append(test_acc)
        model_sizes.append(model_size)
        preservation_losses.append(current_preservation_loss.item())

        if i % 100 == 99:
            print(f"Iteration {i+1}, Test Accuracy: {test_acc:.4f}, Preservation Loss: {current_preservation_loss:.4f}, Model Size: {model_size / (1024 * 1024):.4f} MB")

# After training, evaluate the final model on the test set
final_test_acc = evaluate_accuracy(model, X_test, Y_test)
print(f"Final Test Accuracy: {final_test_acc:.4f}")

# Calculate the final model size
final_model_size = calculate_model_size(model)
print(f"Final Model Size: {final_model_size / (1024 * 1024):.4f} MB")

# Plotting model size over iterations
plt.figure(figsize=(10, 4))
plt.plot(range(len(model_sizes)), [size / (1024 * 1024) for size in model_sizes], label='Model Size (MB)')
plt.xlabel('Evaluation Step')
plt.ylabel('Model Size (MB)')
plt.title('Model Size over Iterations')
plt.legend()
plt.show()

# Plotting preservation loss over iterations
plt.figure(figsize=(10, 4))
plt.plot(range(len(preservation_losses)), preservation_losses, label='Preservation Loss')
plt.xlabel('Evaluation Step')
plt.ylabel('Preservation Loss')
plt.title('Preservation Loss over Iterations')
plt.legend()
plt.show()
 