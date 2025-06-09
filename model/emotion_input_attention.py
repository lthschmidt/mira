import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from utils import config
from model.transformer_mulexpert import Encoder, Decoder
from model.common_layer import Embeddings, LayerNorm

# Определяем устройство автоматически
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmotionInputEncoder(nn.Module):
    """
    Emotion Input Encoder module that handles different types of attention mechanisms
    for integrating emotion information into the Transformer model.
    """

    def __init__(self, emb_dim, hidden_size, num_layers, num_heads,
                 total_key_depth, total_value_depth, filter_size, universal, emo_input_mode):
        """
        Args:
            emb_dim (int): Embedding dimension.
            hidden_size (int): Hidden size of the model.
            num_layers (int): Number of layers in the encoder/decoder.
            num_heads (int): Number of attention heads.
            total_key_depth (int): Dimension of keys in attention.
            total_value_depth (int): Dimension of values in attention.
            filter_size (int): Size of the filter in the feed-forward network.
            universal (bool): Whether to use universal Transformer architecture.
            emo_input_mode (str): Mode for incorporating emotion information ("self_att" or "cross_att").
        """
        super(EmotionInputEncoder, self).__init__()
        self.emo_input_mode = emo_input_mode

        if self.emo_input_mode == "self_att":
            # Use Encoder for self-attention mode
            self.enc = Encoder(
                2 * emb_dim, hidden_size, num_layers, num_heads,
                total_key_depth, total_value_depth, filter_size, universal=universal, input_dropout=config.input_dropout,
            layer_dropout=config.layer_dropout,
            attention_dropout=config.attention_dropout,
            relu_dropout=config.relu_dropout
            )
        elif self.emo_input_mode == "cross_att":
            # Use Decoder for cross-attention mode
            self.enc = Decoder(
                emb_dim, hidden_size, num_layers, num_heads,
                total_key_depth, total_value_depth, filter_size, universal=universal
            )
        else:
            raise ValueError(f"Invalid emotion input mode: {emo_input_mode}.")

    def forward(self, emotion, encoder_outputs, mask_src):
        """
        Forward pass for the EmotionInputEncoder.

        Args:
            emotion (torch.Tensor): Emotion embeddings of shape (batch_size, seq_len, emb_dim).
            encoder_outputs (torch.Tensor): Encoder outputs of shape (batch_size, seq_len, hidden_size).
            mask_src (torch.Tensor): Source mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        if self.emo_input_mode == "self_att":
            # Repeat emotion embeddings to match the sequence length of encoder outputs
            repeat_vals = [-1, encoder_outputs.size(1) // emotion.size(1), -1]
            hidden_state_with_emo = torch.cat([encoder_outputs, emotion.expand(repeat_vals)], dim=-1)
            return self.enc(hidden_state_with_emo, mask_src)
        elif self.emo_input_mode == "cross_att":
            # Use cross-attention between encoder outputs and emotion embeddings
            return self.enc(encoder_outputs, emotion, (None, mask_src))[0]
        else:
            raise ValueError(f"Invalid emotion input mode: {self.emo_input_mode}.")
