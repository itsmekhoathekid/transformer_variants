import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model : int, h : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):   
        d_k = query.shape[-1]

        # batch , h, seqlen, d_k -> batch, h, seqlen, seqlen
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) #  batch, h, seqlen, seqlen
        if dropout is not None:
            attention_scores =  dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # (give it to the next layer), for visualization



    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seqlen, dmodel) - > batch, seq_len, dmodel
        key = self.w_k(k)
        value = self.w_v(v)

        # batch, seqlen, dmodel -> batch, seqlen, h, d_k, -> batch, h, seqlen, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value .shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # batch, h , seqlen, d_k -> batch, seqlen, h, d_k -> batch, seqlen, d_model
        x =  x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # batch , seqlen, dmodel -> batch, seqlen, d_model
        return self.w_o(x)