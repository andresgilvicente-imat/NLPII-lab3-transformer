# test_decoder.py

import pytest
import torch
import torch.nn as nn

from src.decoder import TransformerDecoderLayer, TransformerDecoder
from src.utils import MultiHeadAttention, FeedForward


def _find_submodules_by_type(module, cls):
    found = [child for child in module.children() if isinstance(child, cls)]
    if not found:
        raise AssertionError(f"No submodule of type {cls.__name__} found inside {module}")
    return found


@pytest.mark.order(13)
def test_transformer_decoder_layer_output_shape():
    """Test the output shape of TransformerDecoderLayer's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    intermediate_size = 16

    # Create random input tensors
    x = torch.randn(batch_size, seq_len, d_model)
    enc_output = torch.randn(batch_size, seq_len, d_model)
    tgt_mask = torch.ones(batch_size, seq_len, seq_len)

    # Initialize TransformerDecoderLayer
    decoder_layer = TransformerDecoderLayer(d_model, num_heads, intermediate_size)

    # Forward pass
    output = decoder_layer(x, enc_output, tgt_mask)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )


@pytest.mark.order(14)
def test_decoder_layer_cross_attention_with_different_source_length():
    # The source (encoder) and target (decoder) sequences almost never have
    # the same length. Cross-attention must take its queries from the
    # decoder stream and its keys/values from enc_output, and must not
    # apply the causal target mask (which is tgt_len x tgt_len).
    torch.manual_seed(0)
    b, tgt_len, src_len, d_model = 2, 4, 7, 8
    layer = TransformerDecoderLayer(d_model, 2, 16)
    layer.eval()

    x = torch.randn(b, tgt_len, d_model)
    enc_output = torch.randn(b, src_len, d_model)
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).repeat(b, 1, 1)

    with torch.no_grad():
        output = layer(x, enc_output, tgt_mask)
    assert output.shape == (b, tgt_len, d_model)
    assert torch.isfinite(output).all()


@pytest.mark.order(15)
def test_decoder_layer_attends_to_whole_encoder_output():
    # Every target position (even the first one) must be able to see every
    # source position: changing only the LAST encoder vector must change the
    # output at the FIRST target position. This fails if enc_output is
    # ignored, or if the causal mask is (wrongly) applied in cross-attention.
    torch.manual_seed(0)
    b, seq_len, d_model = 2, 5, 8
    layer = TransformerDecoderLayer(d_model, 2, 16)
    layer.eval()

    x = torch.randn(b, seq_len, d_model)
    enc_output = torch.randn(b, seq_len, d_model)
    enc_output_modified = enc_output.clone()
    enc_output_modified[:, -1, :] += 3.0
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(b, 1, 1)

    with torch.no_grad():
        out = layer(x, enc_output, tgt_mask)
        out_modified = layer(x, enc_output_modified, tgt_mask)

    assert not torch.allclose(out[:, 0, :], out_modified[:, 0, :], atol=1e-5)


@pytest.mark.order(16)
def test_decoder_layer_has_independent_self_and_cross_attention():
    # Self-attention and cross-attention are two different sub-layers with
    # their own weights; reusing the same MultiHeadAttention for both is a
    # bug that does not change any shape.
    layer = TransformerDecoderLayer(8, 2, 16)
    attention_modules = _find_submodules_by_type(layer, MultiHeadAttention)
    assert len(attention_modules) == 2
    assert attention_modules[0] is not attention_modules[1]


@pytest.mark.order(17)
def test_decoder_layer_residual_stream_preserves_input():
    # Zeroing the three sub-layers (self-attention, cross-attention and
    # feed-forward) isolates the residual bookkeeping: with pre-norm
    # residuals the layer must act as the identity.
    torch.manual_seed(0)
    d_model = 8
    layer = TransformerDecoderLayer(d_model, 2, 16)

    for attention_module in _find_submodules_by_type(layer, MultiHeadAttention):
        attention_module.forward = lambda x_q, x_k, x_v, mask=None: torch.zeros_like(x_q)
    for feed_forward_module in _find_submodules_by_type(layer, FeedForward):
        feed_forward_module.forward = lambda h: torch.zeros_like(h)

    b, t = 2, 5
    x = torch.randn(b, t, d_model)
    enc_output = torch.randn(b, 3, d_model)
    tgt_mask = torch.tril(torch.ones(t, t)).unsqueeze(0).repeat(b, 1, 1)
    y = layer(x, enc_output, tgt_mask)
    assert torch.allclose(y, x, atol=1e-6)


@pytest.mark.order(18)
def test_transformer_decoder_output_shape():
    """Test the output shape of TransformerDecoder's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    vocab_size = 10
    max_position_embeddings = 20
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    # Create random input tensors
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    enc_output = torch.randn(batch_size, seq_len, d_model)

    # Initialize TransformerDecoder
    decoder = TransformerDecoder(
        vocab_size,
        max_position_embeddings,
        d_model,
        num_heads,
        intermediate_size,
        num_layers
    )

    # Forward pass
    output = decoder(input_ids, enc_output)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )

    # The stack of decoder layers must actually be used.
    assert not torch.equal(output, decoder.embeddings(input_ids))


@pytest.mark.order(19)
def test_transformer_decoder_layers_match_num_hidden_layers_and_are_independent():
    num_hidden_layers = 3
    decoder = TransformerDecoder(
        vocab_size=10,
        max_position_embeddings=20,
        d_model=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
    )
    layers = list(decoder.layers)
    assert len(layers) == num_hidden_layers
    assert len(set(id(layer) for layer in layers)) == num_hidden_layers


@pytest.mark.order(20)
def test_transformer_decoder_causal_mask_blocks_future_tokens():
    # Changing a later target token must never change the hidden state at an
    # earlier position. Catches a missing, inverted or off-by-one causal mask.
    torch.manual_seed(0)
    vocab_size = 20
    decoder = TransformerDecoder(vocab_size, 20, 8, 2, 16, 2)
    decoder.eval()

    b, tgt_len, src_len = 2, 6, 4
    input_ids = torch.randint(0, vocab_size, (b, tgt_len))
    modified_ids = input_ids.clone()
    modified_ids[:, -1] = (modified_ids[:, -1] + 1) % vocab_size
    enc_output = torch.randn(b, src_len, 8)

    with torch.no_grad():
        out_original = decoder(input_ids, enc_output)
        out_modified = decoder(modified_ids, enc_output)

    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-6)
    assert not torch.allclose(out_original[:, -1, :], out_modified[:, -1, :], atol=1e-6)


@pytest.mark.order(21)
def test_transformer_decoder_uses_encoder_output():
    torch.manual_seed(0)
    vocab_size = 20
    decoder = TransformerDecoder(vocab_size, 20, 8, 2, 16, 2)
    decoder.eval()

    input_ids = torch.randint(0, vocab_size, (2, 4))
    enc_output = torch.randn(2, 6, 8)
    with torch.no_grad():
        out = decoder(input_ids, enc_output)
        out_other_source = decoder(input_ids, torch.randn(2, 6, 8))

    assert not torch.allclose(out, out_other_source, atol=1e-5)
