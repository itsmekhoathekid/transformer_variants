import torch
from model.model import build_transformer
# Giả sử bạn đã định nghĩa sẵn các class:
# InputEmbeddings, PositionalEncoding, MultiHeadAttentionBlock,
# FeedForwardBlock, EncoderBlock, DecoderBlock, Encoder, Decoder, ProjectionLayer, Transformer

# Bước 1: Khởi tạo Transformer
src_vocab_size = 1000
tgt_vocab_size = 1000
src_seq_len = 10
tgt_seq_len = 10
d_model = 512
h = 8
N = 2
dropout = 0.1
d_ff = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
transformer = build_transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    src_seq_len=src_seq_len,
    tgt_seq_len=tgt_seq_len,
    d_model=d_model,
    N=N,
    h=h,
    dropout=dropout,
    d_ff=d_ff
).to(device)

# Bước 2: Tạo dữ liệu giả (random token IDs)
batch_size = 4
src_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)).to("cuda")  # shape: (B, S)
tgt_input = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)).to("cuda")  # shape: (B, T)

# Bước 3: Mask (tùy mô hình bạn định nghĩa mask như thế nào)
# Tạm thời cho None nếu bạn chưa xử lý mask
src_mask = None
tgt_mask = None

# Bước 4: Chạy forward pass
output = transformer(src_input, tgt_input, src_mask, tgt_mask)

print(f"Output shape: {output.shape}")  # kỳ vọng: (batch_size, tgt_seq_len, tgt_vocab_size)
print(output)