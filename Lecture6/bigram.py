import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independant sequences will we process in parallel ?
block_size = 8 # what is the maximum context length for predictions ?
n_embd = 32
max_iters = 3000
learning_rate = 1e-2
device = "cude" if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#----------------


torch.manual_seed(1337)
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from chars to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest eval
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split="train"):
    # generage a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    idxs = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in idxs])
    y = torch.stack([data[i+1:i+block_size+1] for i in idxs])
    x, y = x.to(device), y.to(device)
    return x, y


# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


model = BigramLanguageModel()

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split=split)
            logits, loss = model(X, Y)
            loss[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out