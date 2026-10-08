# test_utils.py

import pytest
import torch
import torch.nn.functional as F
import math
from src.utils import AttentionHead, MultiHeadAttention

@pytest.mark.order(1)
def test_attention_head_output_shape():
    """Test the output shape of AttentionHead's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_k = d_q = d_v = 8

    # Create random input tensors
    x_q = torch.randn(batch_size, seq_len, d_model)
    x_k = torch.randn(batch_size, seq_len, d_model)
    x_v = torch.randn(batch_size, seq_len, d_model)

    # Initialize AttentionHead
    attention_head = AttentionHead(d_model, d_k, d_q, d_v)

    # Forward pass
    output = attention_head(x_q, x_k, x_v)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_v), (
        f"Expected output shape {(batch_size, seq_len, d_v)}, got {output.shape}"
    )

@pytest.mark.order(2)
def test_attention_forward():
    """Test the scaled_dot_product_attention method of AttentionHead with and without mask."""
    import math

    # Create fixed input tensors
    q = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]]])  # Shape: (1, 3, 4)
    k = q.clone()
    v = q.clone()  # Using q as v for simplicity

    # Initialize AttentionHead with identity projections
    attention_head = AttentionHead(d_model=4, d_k=4, d_q=4, d_v=4)
    attention_head.wq = torch.nn.Identity()
    attention_head.wk = torch.nn.Identity()
    attention_head.wv = torch.nn.Identity()

    # Without mask
    output_no_mask = attention_head(q, k, v)  # Using the forward method

    # Expected output without mask
    expected_output_no_mask = torch.tensor([[[0.4519, 0.2741, 0.2741, 0.0000],
                                            [0.2741, 0.4519, 0.2741, 0.0000],
                                            [0.2741, 0.2741, 0.4519, 0.0000]]])

    # Check if the outputs match expected outputs
    assert torch.allclose(output_no_mask, expected_output_no_mask, atol=1e-4), "Output without mask does not match expected output"

    # Now test with mask
    # Create a causal mask that masks out future tokens
    # For position i, tokens at positions > i are masked
    # Mask shape: (1, 3, 3)
    mask = torch.tensor([[[1, 0, 0],
                        [1, 1, 0],
                        [1, 1, 1]]])  # Causal mask

    # Use the scaled_dot_product_attention method with mask
    output_with_mask = attention_head(q, k, v, mask)

    # Expected output with mask
    expected_output_with_mask = torch.tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
                                                [0.3775, 0.6225, 0.0000, 0.0000],
                                                [0.2741, 0.2741, 0.4519, 0.0000]]])

    # Check if the outputs match expected outputs
    assert torch.allclose(output_with_mask, expected_output_with_mask, atol=1e-4), "Output with mask does not match expected output"


@pytest.mark.order(3)
def test_attention_head_cross_attention_uses_each_input_for_its_role():
    # In cross-attention the queries come from one sequence and the keys and
    # values from another one, possibly of a different length. Using x_q for
    # the keys/values (or mixing up x_k and x_v) still produces the right
    # shape when all inputs have the same length, so we use different
    # lengths and compare against a reference computed from the head's own
    # projection layers.
    torch.manual_seed(0)
    b, t_q, t_kv, d_model, d_head = 2, 3, 5, 8, 4
    attention_head = AttentionHead(d_model, d_head, d_head, d_head)

    x_q = torch.randn(b, t_q, d_model)
    x_k = torch.randn(b, t_kv, d_model)
    x_v = torch.randn(b, t_kv, d_model)

    output = attention_head(x_q, x_k, x_v)
    assert output.shape == (b, t_q, d_head)

    q = attention_head.wq(x_q)
    k = attention_head.wk(x_k)
    v = attention_head.wv(x_v)
    expected_weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_head), dim=-1)
    expected = expected_weights @ v
    assert torch.allclose(output, expected, atol=1e-5)


@pytest.mark.order(4)
def test_multihead_attention_output_shape():
    """Test the output shape of MultiHeadAttention's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    # Create random input tensors
    x_q = torch.randn(batch_size, seq_len, d_model)
    x_k = torch.randn(batch_size, seq_len, d_model)
    x_v = torch.randn(batch_size, seq_len, d_model)

    # Initialize MultiHeadAttention
    multihead_attention = MultiHeadAttention(d_model, num_heads)

    # Forward pass
    output = multihead_attention(x_q, x_k, x_v)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )

@pytest.mark.order(5)
def test_multihead_attention_forward():
    """Test the forward method of MultiHeadAttention."""
    # Initialize parameters
    d_model = 4
    num_heads = 2

    # Create fixed input tensors
    x_q = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]]])  # Shape: (1, 3, 4)
    x_k = x_q.clone()
    x_v = x_q.clone()

    # Initialize MultiHeadAttention with identity projections
    multihead_attention = MultiHeadAttention(d_model, num_heads)
    for head in multihead_attention.heads:
        head.wq = torch.nn.Identity()
        head.wk = torch.nn.Identity()
        head.wv = torch.nn.Identity()
    multihead_attention.output_linear = torch.nn.Identity()

    # Forward pass
    output = multihead_attention(x_q, x_k, x_v)

    # Expected outputs. With identity projections every head sees the full
    # 4-dim input, so the scores are scaled by sqrt(4) and the two heads
    # produce identical outputs.
    expected_output = torch.tensor([[[0.4519, 0.2741, 0.2741, 0.0000, 0.4519, 0.2741, 0.2741, 0.0000],
                                    [0.2741, 0.4519, 0.2741, 0.0000, 0.2741, 0.4519, 0.2741, 0.0000],
                                    [0.2741, 0.2741, 0.4519, 0.0000, 0.2741, 0.2741, 0.4519, 0.0000]]])

    # Since output_linear is identity, the final output is the concatenated outputs
    assert torch.allclose(output, expected_output, atol=1e-4), "Output does not match expected output"

    # Create a causal mask that masks out future tokens
    mask = torch.tensor([[[1, 0, 0],
                        [1, 1, 0],
                        [1, 1, 1]]])  # Shape: (1, 3, 3)

    # Forward pass with mask
    output_with_mask = multihead_attention(x_q, x_k, x_v, mask)

    # Expected outputs with mask
    expected_output_with_mask = torch.tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                                                [0.3775, 0.6225, 0.0000, 0.0000, 0.3775, 0.6225, 0.0000, 0.0000],
                                                [0.2741, 0.2741, 0.4519, 0.0000, 0.2741, 0.2741, 0.4519, 0.0000]]])

    # Check if the outputs match expected outputs
    assert torch.allclose(output_with_mask, expected_output_with_mask, atol=1e-4), "Output with mask does not match expected output"


@pytest.mark.order(6)
def test_multihead_attention_heads_are_independent_and_split_d_model():
    # Each head must be its own module (not `[AttentionHead(...)] * n`, which
    # repeats the same instance) and must project to d_model // num_heads,
    # so that concatenating the heads gives back d_model features.
    torch.manual_seed(0)
    d_model, num_heads = 8, 4
    multihead_attention = MultiHeadAttention(d_model, num_heads)
    heads = list(multihead_attention.heads)

    assert len(heads) == num_heads
    assert len(set(id(h) for h in heads)) == num_heads
    assert not torch.equal(heads[0].wq.weight, heads[1].wq.weight)
    for h in heads:
        assert h.wq.out_features == d_model // num_heads
        assert h.wk.out_features == d_model // num_heads
        assert h.wv.out_features == d_model // num_heads


@pytest.mark.order(7)
def test_multihead_attention_cross_attention_matches_reference():
    # Full numeric check of MultiHeadAttention with different query and
    # key/value lengths: every head must receive (x_q, x_k, x_v) in that
    # order, the head outputs must be concatenated in order, and the output
    # projection must be applied afterwards.
    torch.manual_seed(0)
    b, t_q, t_kv, d_model, num_heads = 2, 3, 6, 8, 2
    multihead_attention = MultiHeadAttention(d_model, num_heads)

    x_q = torch.randn(b, t_q, d_model)
    x_k = torch.randn(b, t_kv, d_model)
    x_v = torch.randn(b, t_kv, d_model)

    output = multihead_attention(x_q, x_k, x_v)
    assert output.shape == (b, t_q, d_model)

    head_outputs = []
    for h in multihead_attention.heads:
        q, k, v = h.wq(x_q), h.wk(x_k), h.wv(x_v)
        w = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)), dim=-1)
        head_outputs.append(w @ v)
    expected = multihead_attention.output_linear(torch.cat(head_outputs, dim=-1))
    assert torch.allclose(output, expected, atol=1e-5)
