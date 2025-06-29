import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module): # Self-attention
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        #Note: this dropout randomly prevents some tokens from communicating with each other


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #shape (B,T, head_size)
        q = self.query(x) #shape (B,T, head_size)
        v = self.value(x) #shape (B,T, head_size)

        #compute self-attention scores
        wei = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
        wei *= C**-0.5 #scale by sqrt(d_k) as per paper, so that variance of the wei is 1
        # print(wei.shape, (self.tril[:T,:T]==0).shape)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        #perform weighted aggregation of values
        out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multi-head attention as a collection of heads with concatenated outputs."""
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size*num_heads, n_embd) # combine all head outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ the feed forward network (FFN) in the paper"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        # Note: in the paper (section 3.3) we have d_{model}=512 and d_{ff}=2048.
        # Therefore the inner layer is 4 times the size of the embedding layer
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: comunication (attention) followed by computation (FFN) """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension
        # n_heads : the number of heads we'd like to use
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT2(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.n_embd = n_embd # Smallest GPT2 model uses n_embd=768 per word/token
        self.vocab_size = vocab_size # GPT2 uses 50257
        self.block_size = block_size
        # vocabulary embedding and positional embedding
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # a signal that indicates the order of the words in the sequence to the transformer blocks

        #sequence of attention heads and feed forward layers
        self.blocks = nn.Sequential( *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])

        #one layer normalization layer after transformer blocks
        #and one before linear layer that outputs the vocabulary
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)


    def forward(self, idx):
        """ call the model with idx and targets (training) or without targets (generation)"""

        #idx and targets are both of shape (B,T)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #shape (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) #shape (T,C)
        x = tok_emb + pos_emb #shape (B,T,C)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x) #shape (B,T,C); predict into token vector
        # logits = torch.swapaxes(logits, 1, 2) #shape (B,C,T) to comply with CrossEntropyLoss
        return logits

    def generate(self, inputs, max_new_tokens):
        output = inputs.clone()
        for _ in range(max_new_tokens):
            current_seq_length = output.size(1)
            #### Truncate inputs if it exceeds context_length/block_size
            if current_seq_length >= self.block_size:
                tmp_inputs = output[:, -self.block_size:]
            else:
                tmp_inputs = output
            # print("tmp_inputs:", tmp_inputs.shape, current_seq_length)
            logits = self(tmp_inputs) # shape (B, C, T)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # shape (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            output = torch.cat([output, idx_next], dim=1)
        return output