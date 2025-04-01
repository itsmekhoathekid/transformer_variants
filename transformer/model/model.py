import torch
import torch.nn as nn
import math
from .module import InputEmbeddings, PositionalEncoding, LayerNormalization, FeedForwardBlock, ResidualConnection
from .attention import MultiHeadAttentionBlock


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)



class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # batch ,seqlen, d_model -> batch, seqlen, vocab_size
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048
    ):
        super().__init__()

        # Embedding + Positional Encoding
        self.src_embedding = nn.Sequential(
            InputEmbeddings(d_model, src_vocab_size),
            PositionalEncoding(d_model, src_seq_len, dropout)
        )
        self.tgt_embedding = nn.Sequential(
            InputEmbeddings(d_model, tgt_vocab_size),
            PositionalEncoding(d_model, tgt_seq_len, dropout)
        )

        # Encoder & Decoder Blocks
        encoder_layers = nn.ModuleList([
            EncoderBlock(
                d_model,
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(N)
        ])

        decoder_layers = nn.ModuleList([
            DecoderBlock(
                d_model,
                MultiHeadAttentionBlock(d_model, h, dropout),
                MultiHeadAttentionBlock(d_model, h, dropout),
                FeedForwardBlock(d_model, d_ff, dropout),
                dropout
            ) for _ in range(N)
        ])

        self.encoder = Encoder(d_model, encoder_layers)
        self.decoder = Decoder(d_model, decoder_layers)


        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        """
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)
        
        memory = self.encoder(self.src_pos(src_embedded), src_mask)
        out = self.decoder(self.tgt_pos(tgt_embedded), memory, src_mask, tgt_mask)

        return self.projection_layer(out)
    
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model,
        N=N,
        h=h,
        dropout=dropout,
        d_ff=d_ff
    )

    # # Khởi tạo tham số nếu cần (Xavier init)
    # for param in model.parameters():
    #     if param.dim() > 1:
    #         nn.init.xavier_uniform_(param)
    
    return model




     



