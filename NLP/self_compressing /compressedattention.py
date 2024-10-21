import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from tqdm import trange

# hyperparameters
batch_size = 16
block_size = 32
max_iters = 10000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# Load main text data
with open('/home/mohammad/names.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Load safety data
with open('/home/mohammad/hardest_examples.txt', 'r', encoding='utf-8') as f:
    safety_text = f.read()

# Here are all the unique characters that occur in both texts
chars = sorted(list(set(text + safety_text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]  # encoder function
decode = lambda l: ''.join([itos[i] for i in l])       # decoder function

# Encode the datasets
data = torch.tensor(encode(text), dtype=torch.long)
safety_data = torch.tensor(encode(safety_text), dtype=torch.long)

# Split data into training and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading functions
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_safety_batch():
    ix = torch.randint(len(safety_data) - block_size, (batch_size,))
    x = torch.stack([safety_data[i:i+block_size] for i in ix])
    y = torch.stack([safety_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate loss on training and validation sets
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Define the quantized linear layer
class QLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(QLinear, self).__init__()
        scale = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-scale, scale))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.e = nn.Parameter(torch.full((out_features, 1), -8.))
        self.b = nn.Parameter(torch.full((out_features, 1), 32.))  # Start with 32 bits

    def qbits(self):
        return self.b.relu().sum() * self.weight.shape[1]

    def qweight(self):
        b_rel = self.b.relu()
        min_val = torch.where(b_rel > 0, -2 ** (b_rel - 1), torch.zeros_like(b_rel))
        max_val = torch.where(b_rel > 0, 2 ** (b_rel - 1) - 1, torch.zeros_like(b_rel))
        scaled_weight = 2 ** -self.e * self.weight
        qweight = torch.max(torch.min(scaled_weight, max_val), min_val)
        return qweight

    def forward(self, input):
        qw = self.qweight()
        w = (qw.round() - qw).detach() + qw  # Straight-through estimator
        output = nn.functional.linear(input, 2 ** self.e * w, self.bias)
        return output

# Define the Head class with quantized linear layers
class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = QLinear(n_embd, head_size, bias=False)
        self.query = QLinear(n_embd, head_size, bias=False)
        self.value = QLinear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def qbits(self):
        return self.key.qbits() + self.query.qbits() + self.value.qbits()

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v      # (B,T,head_size)
        return out

# Define MultiHeadAttention with quantized projection
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = QLinear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def qbits(self):
        return sum(h.qbits() for h in self.heads) + self.proj.qbits()

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Define FeedForward with quantized linear layers
class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            QLinear(n_embd, 4 * n_embd),
            nn.ReLU(),
            QLinear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def qbits(self):
        return sum(layer.qbits() for layer in self.net if isinstance(layer, QLinear))

    def forward(self, x):
        return self.net(x)

# Define Transformer Block with quantization
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def qbits(self):
        return self.sa.qbits() + self.ffwd.qbits()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Define the BigramLanguageModel with quantization
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = QLinear(n_embd, vocab_size)

    def qbits(self):
        qbits = 0
        qbits += sum(b.qbits() for b in self.blocks)
        qbits += self.lm_head.qbits()
        return qbits

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)     # (B,T,C)
        x = self.ln_f(x)       # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape      
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] 
            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# Instantiate the model and optimizer
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Compute total weight count for compression calculations
total_weight_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize lists to track model size
model_size_history = []  # To store model size at each eval interval
iteration_history = []   # To store corresponding iterations

# Functions to check and restore zero-bit attention heads
def check_zero_bit_heads():
    for block in model.blocks:
        if isinstance(block, Block):
            for h in block.sa.heads:
                for layer in [h.key, h.query, h.value]:
                    if (layer.b.view(-1) <= 0).any():
                        return True
    return False

def restore_zero_bit_heads(restore_fraction=0.1):
    for block in model.blocks:
        if isinstance(block, Block):
            for h in block.sa.heads:
                for layer in [h.key, h.query, h.value]:
                    b_flat = layer.b.view(-1)
                    zero_bit_indices = (b_flat <= 0).nonzero(as_tuple=False).view(-1)
                    num_restore = int(restore_fraction * len(zero_bit_indices))
                    if num_restore > 0:
                        restore_indices = zero_bit_indices[torch.randperm(len(zero_bit_indices))[:num_restore]]
                        b_flat[restore_indices] = 2.0  # Restore bits to 2

# Variables for tracking
prev_safety_loss = None
safety_loss_increase_threshold = 0.1  # Adjust as needed

# Training loop
for iter in trange(max_iters):
    # Training step
    model.train()
    optimizer.zero_grad()

    # Main training batch
    xb, yb = get_batch('train')
    logits, loss_main = model(xb, yb)
    Q = model.qbits() / total_weight_count

    # Compression regularization weight
    compression_weight = 0.1  # Adjust as needed
    loss = loss_main + compression_weight * Q

    # Safety loss
    xs, ys = get_safety_batch()
    logits_safety, safety_loss = model(xs, ys)
    safety_weight = 0.05  # Adjust as needed
    loss = loss + safety_weight * safety_loss

    loss.backward()
    optimizer.step()

    # Every eval_interval steps, check validation and safety loss, and log model size
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        losses = estimate_loss()
        
        # Calculate current model size in bits and megabytes
        current_qbits = model.qbits()
        current_size_bytes = current_qbits / 8
        current_size_mb = current_size_bytes / 1e6

        # Append to history
        model_size_history.append(current_size_mb)
        iteration_history.append(iter)

        # Log the information
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"safety loss {safety_loss:.4f}, Q: {Q:.4f}, "
              f"Model Size: {current_size_mb:.6f} MB")

        # Restore zero-bit heads if safety loss increases too much
        if prev_safety_loss is not None and (safety_loss - prev_safety_loss) > safety_loss_increase_threshold:
            if check_zero_bit_heads():
                restore_zero_bit_heads(restore_fraction=0.1)
        prev_safety_loss = safety_loss

# Optional: Plot the model size over iterations
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_history, model_size_history, label='Model Size (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Over Training Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
except ImportError:
    print("matplotlib is not installed. Skipping the plot of model size history.")
    # Alternatively, you can save the history to a file or handle it as needed
