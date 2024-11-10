import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-bs', type=str, required=True, help='Please provide a batch size')

args = parser.parse_args()
print(f'batch size: {args.bs}')

batch_size = int(args.bs)
block_size = 128
max_iters = 200 # 3000
learnin_rate = 3e-4
eval_iters = 100 # 300
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2

chars = ""
with open('C:/Users/Oussama/Dataset/Bigram_extraction/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    
vocabulary_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s:[string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head-size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer Block: communictaion followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension (number of features), n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
        

# Bigram Language Model :
class GPTLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        # 1. Embedding + Position Encoding
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 2. how many decoder blocks we're running sequentially
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # repeat Block(n_embd, n_head=n_head) for 4 times : 4 decoders

        # final Linear layer
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, index, targets=None): 
        # logits = self.token_embedding_table(index)
        B, T = index.shape
        
        # index and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop  idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

model = GPTLanguageModel(vocabulary_size)
print('loading model parameters ...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')
m = model.to(device)



while True:
    start_time = time.time()
    prompt = input('Prompt:\n')
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'time taken: {time_taken}.6f')

