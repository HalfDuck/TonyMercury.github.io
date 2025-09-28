"""Transformer-based trajectory forecasting model."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class StaticEmbedding(nn.Module):
    def __init__(self, type_vocab: int, nation_vocab: int, name_vocab: int, dim: int):
        super().__init__()
        self.type_embedding = nn.Embedding(type_vocab, dim)
        self.nation_embedding = nn.Embedding(nation_vocab, dim)
        self.name_embedding = nn.Embedding(name_vocab, dim)
        self.proj = nn.Linear(dim * 3, dim)

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        ship_type, nation, name = static.unbind(dim=-1)
        type_emb = self.type_embedding(ship_type)
        nation_emb = self.nation_embedding(nation)
        name_emb = self.name_embedding(name)
        concat = torch.cat([type_emb, nation_emb, name_emb], dim=-1)
        return self.proj(concat)


class TrajectoryTransformer(nn.Module):
    """Transformer encoder-decoder for multi-step trajectory prediction."""

    def __init__(self, config: ModelConfig, encoder_vocab: Dict[str, int]):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.positional_encoding = PositionalEncoding(config.hidden_dim, dropout=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation=config.activation,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation=config.activation,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.future_queries = nn.Parameter(
            torch.randn(config.predict_steps, config.hidden_dim)
        )

        static_dim = config.hidden_dim
        self.static_embedding = StaticEmbedding(
            type_vocab=encoder_vocab["type"],
            nation_vocab=encoder_vocab["nation"],
            name_vocab=encoder_vocab["name"],
            dim=config.static_feature_dim,
        )
        self.static_proj = nn.Sequential(
            nn.Linear(config.static_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )

        if config.environment_dim:
            self.environment_proj = nn.Linear(config.environment_dim, config.hidden_dim)
        else:
            self.environment_proj = None

        self.position_head = nn.Linear(config.hidden_dim, 2)
        if config.predict_speed:
            self.speed_head = nn.Linear(config.hidden_dim, 1)
        else:
            self.speed_head = None
        if config.predict_heading:
            self.heading_head = nn.Linear(config.hidden_dim, 2)
        else:
            self.heading_head = None

        self.destination_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.num_destination_clusters),
        )

    def forward(
        self,
        history: torch.Tensor,
        static: torch.Tensor,
        environment: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # history: (batch, seq_len, feature_dim)
        x = self.input_proj(history)
        x = self.positional_encoding(x)
        memory = self.encoder(x)

        batch_size = history.shape[0]
        queries = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)

        static_embedding = self.static_embedding(static)
        static_embedding = self.static_proj(static_embedding)
        static_embedding = static_embedding.unsqueeze(1)

        if environment is not None and self.environment_proj is not None:
            env_proj = self.environment_proj(environment)
            env_proj = env_proj.unsqueeze(1)
            memory = memory + env_proj

        memory = memory + static_embedding

        decoded = self.decoder(queries, memory)

        position = self.position_head(decoded)
        outputs: Dict[str, torch.Tensor] = {"delta_xy": position}

        if self.speed_head is not None:
            outputs["speed"] = self.speed_head(decoded).squeeze(-1)
        if self.heading_head is not None:
            heading_raw = self.heading_head(decoded)
            outputs["heading"] = F.normalize(heading_raw, dim=-1)

        pooled = memory.mean(dim=1)
        outputs["destination_logits"] = self.destination_head(pooled)

        return outputs

    def autoregressive_predict(
        self,
        history: torch.Tensor,
        static: torch.Tensor,
        environment: Optional[torch.Tensor] = None,
        step_minutes: int = 10,
        anchor_lat: Optional[torch.Tensor] = None,
        anchor_lon: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict absolute coordinates by integrating predicted displacements."""

        outputs = self.forward(history, static, environment)
        delta_xy = outputs["delta_xy"]  # meters
        cumulative = delta_xy.cumsum(dim=1)
        if anchor_lat is None or anchor_lon is None:
            raise ValueError("anchor_lat and anchor_lon must be provided for absolute predictions")

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = (math.pi / 180.0) * 6378137.0 * torch.cos(torch.deg2rad(anchor_lat))
        lon_offset = cumulative[..., 0] / meters_per_deg_lon
        lat_offset = cumulative[..., 1] / meters_per_deg_lat
        pred_lon = anchor_lon.unsqueeze(1) + lon_offset
        pred_lat = anchor_lat.unsqueeze(1) + lat_offset
        outputs["pred_lon"] = pred_lon
        outputs["pred_lat"] = pred_lat
        return outputs

