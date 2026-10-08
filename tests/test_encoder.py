# test_encoder.py

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.encoder import TransformerEncoderLayer, TransformerEncoder
from src.utils import MultiHeadAttention, FeedForward


def _find_submodules_by_type(module, cls):
    found = [child for child in module.children() if isinstance(child, cls)]
    if not found:
        raise AssertionError(f"No submodule of type {cls.__name__} found inside {module}")
    return found


def _padding_mask(valid_lengths, seq_len):
    # mask[b, i, j] == 1 if query i may attend to key j, i.e. if token j of
    # sequence b is not padding. Shape: (batch_size, seq_len, seq_len).
    valid = torch.zeros(len(valid_lengths), seq_len)
    for i, n in enumerate(valid_lengths):
        valid[i, :n] = 1
    return valid.unsqueeze(1).repeat(1, seq_len, 1)


@pytest.mark.order(8)
def test_transformer_encoder_layer_output_shape():
    """Test the output shape of TransformerEncoderLayer's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    intermediate_size = 16

    # Create random input tensor and mask
    x = torch.randn(batch_size, seq_len, d_model)
    mask = None

    # Initialize TransformerEncoderLayer
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, intermediate_size)

    # Forward pass
    output = encoder_layer(x, mask)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )


@pytest.mark.order(9)
def test_encoder_layer_residual_stream_preserves_input():
    # Zeroing both sub-layers isolates the residual bookkeeping: with
    # pre-norm residuals (x = x + sublayer(norm(x))) the layer must act as
    # the identity. Dropping a residual, adding it to the wrong tensor or
    # normalizing after the sum (post-norm) all break this.
    torch.manual_seed(0)
    d_model, num_heads, intermediate_size = 8, 2, 16
    layer = TransformerEncoderLayer(d_model, num_heads, intermediate_size)

    for attention_module in _find_submodules_by_type(layer, MultiHeadAttention):
        attention_module.forward = lambda x_q, x_k, x_v, mask=None: torch.zeros_like(x_q)
    for feed_forward_module in _find_submodules_by_type(layer, FeedForward):
        feed_forward_module.forward = lambda h: torch.zeros_like(h)

    x = torch.randn(2, 5, d_model)
    y = layer(x, None)
    assert torch.allclose(y, x, atol=1e-6)


@pytest.mark.order(10)
def test_transformer_encoder_output_shape():
    """Test the output shape of TransformerEncoder's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    vocab_size = 10
    max_position_embeddings = 20
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    # Create random input tensor and mask
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = None

    # Initialize TransformerEncoder
    encoder = TransformerEncoder(
        vocab_size,
        max_position_embeddings,
        d_model,
        num_heads,
        intermediate_size,
        num_layers
    )

    # Forward pass
    output = encoder(x, mask)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )

    # The stack of encoder layers must actually be used.
    assert not torch.equal(output, encoder.embeddings(x))


@pytest.mark.order(11)
def test_transformer_encoder_layers_match_num_hidden_layers_and_are_independent():
    num_hidden_layers = 3
    encoder = TransformerEncoder(
        vocab_size=10,
        max_position_embeddings=20,
        d_model=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
    )
    layers = list(encoder.layers)
    assert len(layers) == num_hidden_layers
    assert len(set(id(layer) for layer in layers)) == num_hidden_layers


@pytest.mark.order(12)
def test_transformer_encoder_padding_mask_hides_padding_tokens():
    # The padding mask (batch_size, seq_len, seq_len) must stop real tokens
    # from attending to padding: changing which token sits in a padded
    # position must not change the encoder output at any real position.
    torch.manual_seed(0)
    vocab_size, seq_len = 20, 6
    encoder = TransformerEncoder(vocab_size, 20, 8, 2, 16, 2)
    encoder.eval()

    valid_lengths = [4, 2]
    mask = _padding_mask(valid_lengths, seq_len)

    x = torch.randint(1, vocab_size, (2, seq_len))
    x_other_padding = x.clone()
    for i, n in enumerate(valid_lengths):
        x_other_padding[i, n:] = (x[i, n:] + 7) % vocab_size

    with torch.no_grad():
        out = encoder(x, mask)
        out_other_padding = encoder(x_other_padding, mask)

    for i, n in enumerate(valid_lengths):
        assert torch.isfinite(out[i, :n]).all()
        assert torch.allclose(out[i, :n], out_other_padding[i, :n], atol=1e-5)

    # Sanity check: without the mask, the padding tokens DO leak into the
    # real positions, so the check above is meaningful.
    with torch.no_grad():
        out_no_mask = encoder(x, None)
        out_no_mask_other_padding = encoder(x_other_padding, None)
    assert not torch.allclose(out_no_mask[0, :4], out_no_mask_other_padding[0, :4], atol=1e-5)
