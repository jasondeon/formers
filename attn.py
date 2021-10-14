import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.parameter import Parameter

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.dropout = dropout
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == self.embed_dim, "embed_dim must be divisible by nhead"

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, q, attn_mask=None):
        # set up shape vars
        bsz, tgt_len, embed_dim = q.shape
        assert embed_dim == self.embed_dim, f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        
        # compute in-projection
        q, k, v = F.linear(q, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        # combine head indices with batch indices
        q = q.transpose(0,1).contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0,1)
        k = k.transpose(0,1).contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0,1)
        v = v.transpose(0,1).contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0,1)
        
        # adjust dropout probability
        dropout_p = self.dropout if self.training else 0.0
        
        # calculate attention and out projection
        attn_output = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0,1).contiguous().view(tgt_len, bsz, embed_dim).transpose(0,1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class RelativeSelfAttention(nn.Module):
    def __init__(self, embed_dim, nhead, k, dropout=0.):
        super(RelativeSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.k = k
        self.dropout = dropout
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == self.embed_dim, "embed_dim must be divisible by nhead"

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rel_pe = Parameter(torch.empty((2*self.k+1, self.head_dim)))

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        xavier_uniform_(self.out_proj.weight)
        xavier_uniform_(self.rel_pe)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, q, attn_mask=None):
        # set up shape vars
        bsz, tgt_len, embed_dim = q.shape
        assert embed_dim == self.embed_dim, f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        
        # compute in-projection
        q, k, v = F.linear(q, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        # combine head indices with batch indices
        q = q.transpose(0,1).contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0,1)
        k = k.transpose(0,1).contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0,1)
        v = v.transpose(0,1).contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0,1)
        
        # prepare A (relative position representations)
        a = torch.zeros((tgt_len, tgt_len, self.head_dim))
        clipped_rel_pe = torch.cat((self.rel_pe[0].repeat(tgt_len-self.k-1,1), self.rel_pe, self.rel_pe[-1].repeat(tgt_len-self.k-1,1)))
        for i in range(tgt_len):
            a[i,:,:] = clipped_rel_pe[tgt_len-i-1 : 2*tgt_len-i-1]
        
        # adjust dropout probability
        dropout_p = self.dropout if self.training else 0.0
        
        # calculate attention and out projection
        attn_output = _scaled_dot_product_attention_relative(q, k, v, a, attn_mask, dropout_p)
        attn_output = attn_output.transpose(0,1).contiguous().view(tgt_len, bsz, embed_dim).transpose(0,1)
        attn_output = self.out_proj(attn_output)
        return attn_output


def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    B, Nt, E = q.shape
    q = q / math.sqrt(E) # E is head dim    
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    z = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        z += attn_mask
    out = F.softmax(z, dim=-1)
    if dropout_p > 0.0:
        out = F.dropout(out, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    out = torch.bmm(out, v)
    return out


def _scaled_dot_product_attention_relative(q, k, v, a, attn_mask=None, dropout_p=0.0):
    B, Nt, E = q.shape
    q = q / math.sqrt(E) # E is head dim
    # QK^T : (B, N, E) x (B, E, N) -> (B, N, N)
    # QA^T : (N, B, E) x (N, E, N) -> (N, B, N)
    z = torch.bmm(q, k.transpose(-2, -1)) + torch.bmm(q.transpose(0,1), a.transpose(-2, -1)).transpose(0, 1)
    if attn_mask is not None:
        z += attn_mask
    out = F.softmax(z, dim=-1)
    if dropout_p > 0.0:
        out = F.dropout(out, p=dropout_p)
    # (B, N, N) x (B, N, E) -> (B, N, E)
    out = torch.bmm(out, v)
    return out