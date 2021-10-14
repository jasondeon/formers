import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.parameter import Parameter
from attn import SelfAttention, RelativeSelfAttention


class LayerStack(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(LayerStack, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        return output


class BaseDecoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward=2048, dropout=0.,
                 layer_norm_eps=1e-5, device=None, dtype=None):
        super(BaseDecoderLayer, self).__init__()
        self.self_attn = SelfAttention(embed_dim, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.act = F.relu
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)
        constant_(self.linear1.bias, 0.)
        constant_(self.linear2.bias, 0.)
    
    def forward(self, src):
        # Self-attention sublayer
        # Decoder is auto-regressive, so don't attend to future tokens in self_attn
        attn_mask = torch.triu(torch.ones(src.size(1), src.size(1)), diagonal=1).bool()
        attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf")).unsqueeze(0)
        src2 = self.self_attn(src, attn_mask=attn_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward sublayer
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class RelativeDecoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, k, dim_feedforward=2048, dropout=0.,
                 layer_norm_eps=1e-5, device=None, dtype=None):
        super(RelativeDecoderLayer, self).__init__()
        self.self_attn = RelativeSelfAttention(embed_dim, nhead, k, dropout=dropout)
        
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.act = F.relu
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)
        constant_(self.linear1.bias, 0.)
        constant_(self.linear2.bias, 0.)
    
    def forward(self, src):
        # Self-attention sublayer
        # Decoder is auto-regressive, so don't attend to future tokens in self_attn
        attn_mask = torch.triu(torch.ones(src.size(1), src.size(1)), diagonal=1).bool()
        attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf")).unsqueeze(0)
        src2 = self.self_attn(src, attn_mask=attn_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward sublayer
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src