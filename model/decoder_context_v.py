import torch
import torch.nn as nn
from utils import config
import pprint
import numpy as np
from model.common_layer import EncoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask, _gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask, \
    get_input_from_batch, get_output_from_batch, top_k_top_p_filtering, evaluate, count_parameters, make_infinite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderLayerContextV(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        super(DecoderLayerContextV, self).__init__()

        self.multi_head_attention_dec = MultiHeadAttention(
            hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout
        )

        self.multi_head_attention_enc_dec = MultiHeadAttention(
            hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, None, attention_dropout
        )

        self.positionwise_feed_forward = PositionwiseFeedForward(
            hidden_size, filter_size, hidden_size, layer_config='cc', padding='left', dropout=relu_dropout
        )

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple): A tuple containing:
                - x (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, hidden_size).
                - encoder_outputs (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, hidden_size).
                - v (torch.Tensor): Additional context tensor of shape (batch_size, seq_len, hidden_size).
                - attention_weight: Unused placeholder.
                - mask (tuple): A tuple containing:
                    - mask_src (torch.Tensor): Source mask tensor.
                    - dec_mask (torch.Tensor): Decoder mask tensor.

        Returns:
            tuple: A tuple containing:
                - y (torch.Tensor): Output tensor of shape (batch_size, seq_len, hidden_size).
                - encoder_outputs (torch.Tensor): Encoder output tensor.
                - attention_weight: Attention weights.
                - mask (tuple): Source and decoder masks.
        """
        x, encoder_outputs, v, attention_weight, mask = inputs
        mask_src, dec_mask = mask

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)

        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, v, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        return y, encoder_outputs, attention_weight, mask


class DecoderContextV(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Decoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            universal: Whether to use universal Transformer architecture
        """
        super(DecoderContextV, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size).to(device)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size).to(device)

        self.mask = _get_attn_subsequent_mask(max_length).to(device)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout
        )

        if self.universal:
            self.dec = DecoderLayerContextV(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayerContextV(*params) for _ in range(num_layers)])\

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(0.2)

    def forward(self, inputs, encoder_output, v, mask):
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_size).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, hidden_size).
            v (torch.Tensor): Additional context tensor of shape (batch_size, seq_len, hidden_size).
            mask (tuple): A tuple containing:
                - mask_src (torch.Tensor): Source mask tensor.
                - mask_trg (torch.Tensor): Target mask tensor.

        Returns:
            tuple: A tuple containing:
                - y (torch.Tensor): Output tensor of shape (batch_size, seq_len, hidden_size).
                - attn_dist: Attention distribution.
        """
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)

        # Add input dropout
        x = self.input_dropout(inputs)
        if not config.project:
            x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                raise NotImplementedError("ACT is not implemented in this version.")
            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, v, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, v, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)

        return y, attn_dist
