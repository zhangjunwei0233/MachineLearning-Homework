from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def make_length_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max().item())
    rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return rng < lengths.unsqueeze(1)


class GatedAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        mask = make_length_mask(lengths, x.size(1))
        scores = self.v(torch.tanh(self.proj(x))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # (B, D)
        return context


class ResidualRNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        self.proj = None
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        if self.output_dim != input_dim:
            self.proj = nn.Linear(input_dim, self.output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.layer_norm(out)
        residual = x
        if self.proj is not None:
            residual = self.proj(residual)
        out = out + residual[:, : out.size(1), :]
        return out, h


class RNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(ResidualRNNLayer(in_dim, hidden_dim, bidirectional=True))
            in_dim = hidden_dim * 2
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_last_fw = None
        h_last_bw = None
        out = x
        for layer in self.layers:
            out = self.dropout(out)
            out, h = layer(out, lengths)
            # h shape: (num_layers*num_directions, B, H) but here num_layers=1 per block
            h_last_fw = h[-2] if h.size(0) == 2 else h[-1]
            h_last_bw = h[-1] if h.size(0) == 2 else torch.zeros_like(h_last_fw)
        return out, h_last_fw, h_last_bw


class SentimentRNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 5,
        embed_dropout: float = 0.1,
        encoder_dropout: float = 0.1,
        attn_hidden: int = 128,
        head_hidden: int = 256,
        pretrained_embeds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeds is not None:
            with torch.no_grad():
                self.embed.weight[: pretrained_embeds.size(0), : pretrained_embeds.size(1)].copy_(pretrained_embeds)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.encoder = RNNEncoder(embed_dim, hidden_dim, num_layers=num_layers, dropout=encoder_dropout)
        encoder_out_dim = self.encoder.output_dim
        self.attn = GatedAttentionPool(encoder_out_dim, attn_hidden)
        pooled_dim = encoder_out_dim * 4  # attn + last fw + last bw + mean/max
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = make_length_mask(lengths, input_ids.size(1))
        embeds = self.embed(input_ids)
        embeds = self.embed_dropout(embeds)
        enc_out, h_fw, h_bw = self.encoder(embeds, lengths)

        mean_pool = (enc_out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        enc_out_masked = enc_out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = enc_out_masked.max(dim=1).values
        attn_ctx = self.attn(enc_out, lengths)

        feats = torch.cat([attn_ctx, h_fw, h_bw, mean_pool, max_pool], dim=1)
        logits = self.proj(feats)
        return logits
