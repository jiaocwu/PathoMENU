import torch
import torch.nn as nn
import torch.nn.functional as F



class SequenceEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.5):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = nn.Parameter(positional_encoding(1000, self.d_model,device=torch.device('cpu')))
        self.enc_layers = nn.ModuleList([MultiHeadEncoderLayer(d_model, num_heads, dff, rate)
                                        for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x

def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(torch.tensor(10000.0), (2 * (i // 2)) / torch.tensor(d_model,dtype=torch.float32))
    return pos * angle_rates

def positional_encoding(position, d_model,device):
    angle_rads = get_angles(torch.arange(position,device=device).unsqueeze(-1),
                            torch.arange(d_model,device=device).unsqueeze(0),
                            d_model)

    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads.unsqueeze(0)
    return pos_encoding

class MultiHeadEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
      x = x.view(batch_size, -1, self.num_heads, self.depth)
      return x.transpose(1, 2)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(1, 2)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-1,-2))
    dk = k.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk,dtype=torch.float32,device=q.device))
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights
