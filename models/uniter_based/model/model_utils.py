import torch
import math

class ObjPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model = 128, dropout = 0.1, max_len = 10002):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: (#batch, #seq_length)
        """
        pos = x * 5
        pos += 5001
        pos = torch.round(pos.float()).long()
        out = self.pe[pos]

        # padding pos returns a 0 vertor
        out[[x==0]] = 0
        return self.dropout(out) # (batch, len_seq, 128)

def make_KBid_emb_init(emb_mat, KBid_init_path='./processed/KB_emb.pt'):
    KBid_emb_init = torch.load(KBid_init_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for key, val in KBid_emb_init.items():
        emb_mat[key,:] = val
    return emb_mat