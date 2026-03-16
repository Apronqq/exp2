import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from layers.Embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout=dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, attn_mask=None):
        enc_out = self.embedding(x_enc, x_mark_enc)
        enc_out, series, prior, sigmas = self.encoder(enc_out, attn_mask=attn_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = bool(getattr(configs, 'output_attention', False))
        self.model = AnomalyTransformer(
            win_size=configs.seq_len,
            enc_in=configs.enc_in,
            c_out=configs.c_out,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.e_layers,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            output_attention=self.output_attention
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, attn_mask=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, attn_mask)
