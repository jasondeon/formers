import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_
from decoder import BaseDecoderLayer, RelativeDecoderLayer, LayerStack

class BaseTransformer(nn.Module):
    # Baseline Transformer (Decoder only)
    def __init__(self,
                 seq_len,
                 embed_dim,
                 nhead,
                 feedforward,
                 layers,
                 dropout):
        super(BaseTransformer, self).__init__()
        
        self.seq_len     = seq_len
        self.embed_dim   = embed_dim
        self.nhead       = nhead
        self.feedforward = feedforward
        self.layers      = layers
        self.dropout     = dropout
        
        self.embed       = nn.Embedding(128+128+100+32, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, max_len=self.seq_len)
        dec_layer        = BaseDecoderLayer(self.embed_dim, self.nhead, self.feedforward, self.dropout)
        dec_norm         = nn.LayerNorm(self.embed_dim)
        self.dec         = LayerStack(dec_layer, self.layers, norm=dec_norm)
        self.fc          = nn.Linear(self.embed_dim, 128+128+100+32)
        
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.embed.weight)
        xavier_uniform_(self.fc.weight)
        constant_(self.fc.bias, 0.)

    def forward(self, inp):
        out = self.embed(inp) * math.sqrt(self.embed_dim)
        out = self.pos_encoder(out)
        out = self.dec(out)
        out = self.fc(out)
        return out


class RelativeTransformer(nn.Module):
    # Relative Transformer (Decoder only)
    def __init__(self,
                 seq_len,
                 embed_dim,
                 nhead,
                 k,
                 feedforward,
                 layers,
                 dropout):
        super(RelativeTransformer, self).__init__()
        
        self.seq_len     = seq_len
        self.embed_dim   = embed_dim
        self.nhead       = nhead
        self.k           = k
        self.feedforward = feedforward
        self.layers      = layers
        self.dropout     = dropout
        
        self.embed       = nn.Embedding(128+128+100+32, self.embed_dim)
        dec_layer        = RelativeDecoderLayer(self.embed_dim, self.nhead, self.k, self.feedforward, self.dropout)
        dec_norm         = nn.LayerNorm(self.embed_dim)
        self.dec         = LayerStack(dec_layer, self.layers, norm=dec_norm)
        self.fc          = nn.Linear(self.embed_dim, 128+128+100+32)
        
        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.embed.weight)
        xavier_uniform_(self.fc.weight)
        constant_(self.fc.bias, 0.)

    def forward(self, inp):
        out = self.embed(inp) * math.sqrt(self.embed_dim)
        out = self.dec(out)
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term) # All even indices of the embedding dim.
        pe[:, 1::2] = torch.cos(position * div_term) # All odd indices of the embedding dim.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)