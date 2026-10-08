# test_transformer.py

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer import Transformer
from src.decoder import TransformerDecoder
from src.encoder import TransformerEncoder


def _build_transformer(vocab_size=1000, d_model=512, num_heads=8, intermediate_size=2048, num_layers=6,
                       max_position_embeddings=50):
    return Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_enc_position_embeddings=max_position_embeddings,
        max_dec_position_embeddings=max_position_embeddings,
        enc_d_model=d_model,
        dec_d_model=d_model,
        num_attention_heads=num_heads,
        enc_intermediate_size=intermediate_size,
        dec_intermediate_size=intermediate_size,
        num_enc_hidden_layers=num_layers,
        num_dec_hidden_layers=num_layers
    )


# -------- Mock language model for the decoding strategies --------
#
# The encoder, decoder and output layer are replaced by deterministic mocks
# so that the expected output of every decoding strategy can be worked out
# by hand, independently of how many times (or in which order) the
# implementation calls the decoder / output layer:
#
#   - the encoder returns zeros,
#   - the decoder returns the one-hot encoding of each target token
#     (so the "hidden state" of a position identifies its token, and the
#     cosine similarity between two hidden states is 1 if the tokens are
#     equal and 0 otherwise),
#   - the output layer maps the one-hot of the current token to the logits
#     of the next token, log(next_token_probs[current_token]).
#
# This is a bigram language model: P(y_t | y_<t, x) = next_token_probs[y_{t-1}].

VOCAB_SIZE = 8
SOS = 0
EOS = 7


class _MockEncoder(nn.Module):
    def forward(self, src_input, *args, **kwargs):
        return torch.zeros(src_input.size(0), src_input.size(1), VOCAB_SIZE)


class _MockDecoder(nn.Module):
    def forward(self, tgt_input, *args, **kwargs):
        return F.one_hot(tgt_input.long(), VOCAB_SIZE).float()


class _MockOutputLinear(nn.Module):
    def __init__(self, next_token_probs):
        super().__init__()
        self.register_buffer("log_probs", torch.log(next_token_probs))

    def forward(self, hidden):
        return hidden @ self.log_probs


def _bigram_model(next_token_probs):
    next_token_probs = torch.as_tensor(next_token_probs, dtype=torch.float)
    next_token_probs = next_token_probs / next_token_probs.sum(dim=-1, keepdim=True)
    model = _build_transformer(vocab_size=VOCAB_SIZE, d_model=4, num_heads=2, intermediate_size=8, num_layers=1)
    model.encoder = _MockEncoder()
    model.decoder = _MockDecoder()
    model.output_linear = _MockOutputLinear(next_token_probs)
    model.eval()
    return model


def _row(peaks):
    # Distribution over the vocabulary with the given {token: prob} peaks and
    # the remaining mass spread uniformly over the other tokens.
    rest = [t for t in range(VOCAB_SIZE) if t not in peaks]
    remaining = 1.0 - sum(peaks.values())
    return [peaks.get(t, remaining / len(rest)) for t in range(VOCAB_SIZE)]


def _same_row_for_every_token(peaks):
    return [_row(peaks) for _ in range(VOCAB_SIZE)]


def _first_token_frequencies(model, n_samples, **generate_kwargs):
    src_input = torch.zeros(n_samples, 3, dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(src_input, max_length=1, SOS_token=SOS, EOS_token=EOS, **generate_kwargs)
    assert generated.shape == (n_samples, 1)
    counts = torch.bincount(generated[:, 0], minlength=VOCAB_SIZE).float()
    return counts / n_samples


# -------- Transformer: architecture --------

@pytest.mark.order(22)
def test_transformer_initialization():
    """Test if the Transformer model initializes correctly."""
    model = _build_transformer()

    assert model.encoder is not None
    assert model.decoder is not None
    assert model.output_linear is not None
    assert isinstance(model.encoder, TransformerEncoder)
    assert isinstance(model.decoder, TransformerDecoder)

@pytest.mark.order(23)
def test_transformer_forward_pass_output_shape():
    """Test the forward pass of the Transformer model."""
    tgt_vocab_size = 1000
    model = _build_transformer(vocab_size=tgt_vocab_size)

    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12

    src_input = torch.randint(0, 1000, (batch_size, src_seq_len))
    tgt_input = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    attn_mask = None  # Assuming no mask for simplicity

    output = model(src_input, tgt_input, attn_mask)

    assert output.shape == (batch_size, tgt_seq_len, tgt_vocab_size), "Output shape mismatch"


@pytest.mark.order(24)
def test_transformer_forward_connects_encoder_decoder_and_mask():
    # The source sentence must reach the logits through the encoder output,
    # the padding mask must be handed to the encoder, and the decoder side
    # must stay causal.
    torch.manual_seed(0)
    vocab_size = 30
    model = _build_transformer(vocab_size=vocab_size, d_model=8, num_heads=2, intermediate_size=16, num_layers=2)
    model.eval()

    src = torch.randint(0, vocab_size, (2, 5))
    other_src = torch.randint(0, vocab_size, (2, 5))
    tgt = torch.randint(0, vocab_size, (2, 4))

    with torch.no_grad():
        logits = model(src, tgt)
        logits_other_src = model(other_src, tgt)
    assert not torch.allclose(logits, logits_other_src, atol=1e-5)

    modified_tgt = tgt.clone()
    modified_tgt[:, -1] = (modified_tgt[:, -1] + 1) % vocab_size
    with torch.no_grad():
        logits_modified_tgt = model(src, modified_tgt)
    assert torch.allclose(logits[:, :-1], logits_modified_tgt[:, :-1], atol=1e-6)

    received = []
    original_encoder_forward = model.encoder.forward

    def spy(*args, **kwargs):
        received.extend(list(args) + list(kwargs.values()))
        return original_encoder_forward(*args, **kwargs)

    model.encoder.forward = spy
    attn_mask = torch.ones(2, 5, 5)
    with torch.no_grad():
        model(src, tgt, attn_mask)
    assert any(a is attn_mask for a in received), "attn_mask is not passed to the encoder"


# -------- generate(): output shapes on a real (untrained) model --------

@pytest.mark.order(25)
def test_transformer_generate_greedy_output_shape():
    """Test the generate method using greedy decoding."""
    model = _build_transformer()

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, 1000, (batch_size, src_seq_len))

    generated_sequence = model.generate(src_input, max_length=15, decoding_strategy='greedy')

    assert generated_sequence.shape[0] == batch_size, "Batch size mismatch in generated sequences"
    assert generated_sequence.shape[1] <= 15, "Generated sequence length exceeds max_length"

@pytest.mark.order(26)
def test_transformer_generate_beam_search_output_shape():
    """Test the generate method using beam search decoding."""
    model = _build_transformer()

    batch_size = 1  # Beam search currently supports batch_size=1
    src_seq_len = 10
    src_input = torch.randint(0, 1000, (batch_size, src_seq_len))

    generated_sequence = model.generate(src_input, max_length=15, decoding_strategy='beam_search', beam_size=3)

    assert generated_sequence.shape[0] == batch_size, "Batch size mismatch in generated sequences"
    assert generated_sequence.shape[1] <= 15, "Generated sequence length exceeds max_length"

@pytest.mark.order(27)
def test_transformer_generate_sampling_output_shape():
    """Test the generate method using multinomial sampling."""
    model = _build_transformer()

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, 1000, (batch_size, src_seq_len))

    generated_sequence = model.generate(src_input, max_length=15, decoding_strategy='sampling', temperature=1.0)

    assert generated_sequence.shape[0] == batch_size, "Batch size mismatch in generated sequences"
    assert generated_sequence.shape[1] <= 15, "Generated sequence length exceeds max_length"

@pytest.mark.order(28)
def test_transformer_generate_top_k_output_shape():
    """Test the generate method using top-k sampling."""
    model = _build_transformer()

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, 1000, (batch_size, src_seq_len))

    generated_sequence = model.generate(src_input, max_length=15, decoding_strategy='top_k', k=10)

    assert generated_sequence.shape[0] == batch_size, "Batch size mismatch in generated sequences"
    assert generated_sequence.shape[1] <= 15, "Generated sequence length exceeds max_length"

@pytest.mark.order(29)
def test_transformer_generate_top_p_output_shape():
    """Test the generate method using top-p sampling."""
    model = _build_transformer()

    batch_size = 2
    src_seq_len = 10
    src_input = torch.randint(0, 1000, (batch_size, src_seq_len))

    generated_sequence = model.generate(src_input, max_length=15, decoding_strategy='top_p', p=0.9)

    assert generated_sequence.shape[0] == batch_size, "Batch size mismatch in generated sequences"
    assert generated_sequence.shape[1] <= 15, "Generated sequence length exceeds max_length"

@pytest.mark.order(30)
def test_transformer_generate_contrastive_output_shape():
    """Test the generate method using contrastive decoding."""
    model = _build_transformer()

    batch_size = 1  # Contrastive decoding currently supports batch_size=1
    src_seq_len = 10
    src_input = torch.randint(0, 1000, (batch_size, src_seq_len))

    generated_sequence = model.generate(src_input, max_length=15, decoding_strategy='contrastive', k=5, alpha=0.6)

    assert generated_sequence.shape[0] == batch_size, "Batch size mismatch in generated sequences"
    assert generated_sequence.shape[1] <= 15, "Generated sequence length exceeds max_length"

@pytest.mark.order(31)
def test_transformer_generate_eos_termination():
    """Test that generation stops when EOS token is generated."""
    tgt_vocab_size = 1000
    model = _build_transformer(vocab_size=tgt_vocab_size, d_model=4, num_heads=2, intermediate_size=16, num_layers=2)

    batch_size = 1
    src_seq_len = 5
    src_input = torch.randint(0, tgt_vocab_size, (batch_size, src_seq_len))

    # Mock the decoder to produce EOS token immediately
    def mock_decoder(tgt_input, enc_output):
        batch_size = tgt_input.size(0)
        seq_len = tgt_input.size(1)
        vocab_size = tgt_vocab_size
        # Generate logits that will result in EOS token
        logits = torch.full((batch_size, seq_len, vocab_size), float('-inf'))
        logits[:, -1, 3] = 0  # Assuming EOS_token = 3
        return logits

    # Replace the decoder's forward method with the mock
    model.decoder.forward = mock_decoder
    model.output_linear = torch.nn.Identity()

    generated_sequence = model.generate(src_input, max_length=15, decoding_strategy='greedy', EOS_token=3)

    assert generated_sequence.shape[1] == 1, "Generation did not stop at EOS token"


# -------- generate(): decoding strategies on the mock bigram model --------

@pytest.mark.order(32)
def test_greedy_decoding():
    # Greedy must follow the most likely next token given the LAST generated
    # token (0 -> 1 -> 2 -> ...), must not include the SOS token in the
    # output and must work for a batch.
    next_token_probs = [_row({(t + 1) % EOS: 0.6}) for t in range(VOCAB_SIZE)]
    model = _bigram_model(next_token_probs)

    src_input = torch.zeros(2, 3, dtype=torch.long)
    with torch.no_grad():
        generated_sequence = model.generate(src_input, max_length=5, decoding_strategy='greedy',
                                            SOS_token=SOS, EOS_token=EOS)

    assert generated_sequence.tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], \
        "Greedy decoding did not produce expected tokens."


# Bigram model where greedy and beam search disagree:
#   from SOS: 1 (0.5) or 2 (0.4)
#   from 1:   3 (0.32), 4 (0.30), 5 (0.28)   -> best path through 1 is 0.5 * 0.32 = 0.16
#   from 2:   6 (0.9)                        -> path through 2 is 0.4 * 0.9  = 0.36
_BEAM_PROBS = [_row({}) for _ in range(VOCAB_SIZE)]
_BEAM_PROBS[SOS] = _row({1: 0.5, 2: 0.4})
_BEAM_PROBS[1] = _row({3: 0.32, 4: 0.30, 5: 0.28})
_BEAM_PROBS[2] = _row({6: 0.9})


@pytest.mark.order(33)
def test_beam_search_decoding():
    model = _bigram_model(_BEAM_PROBS)
    src_input = torch.zeros(1, 3, dtype=torch.long)

    with torch.no_grad():
        greedy = model.generate(src_input, max_length=2, decoding_strategy='greedy', SOS_token=SOS, EOS_token=EOS)
        beam_1 = model.generate(src_input, max_length=2, decoding_strategy='beam_search', beam_size=1,
                                SOS_token=SOS, EOS_token=EOS)
        beam_2 = model.generate(src_input, max_length=2, decoding_strategy='beam_search', beam_size=2,
                                SOS_token=SOS, EOS_token=EOS)
        beam_3 = model.generate(src_input, max_length=2, decoding_strategy='beam_search', beam_size=3,
                                SOS_token=SOS, EOS_token=EOS)

    assert greedy.tolist() == [[1, 3]]
    assert beam_1.tolist() == [[1, 3]], "Beam search with beam_size=1 must behave like greedy decoding."
    assert beam_2.tolist() == [[2, 6]], "Beam search did not return the most likely sequence."
    assert beam_3.tolist() == [[2, 6]], "Beam search did not return the most likely sequence."


# Same next-token distribution after every token, so the sampling tests only
# need the first generated token.
_SAMPLING_PROBS = _same_row_for_every_token({1: 0.35, 2: 0.25, 3: 0.2, 4: 0.1, 5: 0.06})


def _expected(probs_by_token):
    expected = torch.zeros(VOCAB_SIZE)
    for t, p in probs_by_token.items():
        expected[t] = p
    return expected / expected.sum()


@pytest.mark.order(34)
def test_sampling_decoding():
    torch.manual_seed(0)
    model = _bigram_model(_SAMPLING_PROBS)
    probs = torch.tensor(_SAMPLING_PROBS[0])

    for temperature in (1.0, 0.5, 2.0):
        frequencies = _first_token_frequencies(model, 4000, decoding_strategy='sampling', temperature=temperature)
        expected = F.softmax(torch.log(probs) / temperature, dim=-1)
        assert torch.allclose(frequencies, expected, atol=0.035), (
            f"Sampled token frequencies {frequencies.tolist()} do not match softmax(logits / T) "
            f"{expected.tolist()} for temperature={temperature}"
        )


@pytest.mark.order(35)
def test_top_k_sampling_decoding():
    torch.manual_seed(0)
    model = _bigram_model(_SAMPLING_PROBS)

    frequencies = _first_token_frequencies(model, 4000, decoding_strategy='top_k', k=3)
    expected = _expected({1: 0.35, 2: 0.25, 3: 0.2})
    assert frequencies[expected == 0].sum() == 0, "Top-k sampled a token outside the k most likely ones."
    assert torch.allclose(frequencies, expected, atol=0.035), (
        f"Top-k frequencies {frequencies.tolist()} do not match the renormalized top-k distribution {expected.tolist()}"
    )

    frequencies = _first_token_frequencies(model, 2000, decoding_strategy='top_k', k=1)
    assert frequencies[1] == 1.0, "Top-k with k=1 must always pick the most likely token."


@pytest.mark.order(36)
def test_top_p_sampling_decoding():
    # Sorted probabilities: 0.35, 0.25, 0.2, ... -> cumulative 0.35, 0.60, 0.80, ...
    # The nucleus is the SMALLEST set of most likely tokens whose cumulative
    # probability exceeds p.
    torch.manual_seed(0)
    model = _bigram_model(_SAMPLING_PROBS)

    # p = 0.5: {1, 2} (0.60 > 0.5; {1} alone is not enough)
    frequencies = _first_token_frequencies(model, 4000, decoding_strategy='top_p', p=0.5)
    expected = _expected({1: 0.35, 2: 0.25})
    assert frequencies[expected == 0].sum() == 0, "Top-p sampled a token outside the nucleus."
    assert torch.allclose(frequencies, expected, atol=0.035), (
        f"Top-p frequencies {frequencies.tolist()} do not match the renormalized nucleus {expected.tolist()}"
    )

    # p = 0.7: {1, 2, 3}
    frequencies = _first_token_frequencies(model, 4000, decoding_strategy='top_p', p=0.7)
    expected = _expected({1: 0.35, 2: 0.25, 3: 0.2})
    assert frequencies[expected == 0].sum() == 0, "Top-p sampled a token outside the nucleus."
    assert torch.allclose(frequencies, expected, atol=0.035)

    # p = 0.3: the most likely token alone already exceeds p.
    frequencies = _first_token_frequencies(model, 2000, decoding_strategy='top_p', p=0.3)
    assert frequencies[1] == 1.0, "Top-p must keep the most likely token even when it alone exceeds p."


# Same distribution after every token: 1 (0.8), 2 (0.1), 3 (0.05), rest 0.01.
_CONTRASTIVE_PROBS = _same_row_for_every_token({1: 0.8, 2: 0.1, 3: 0.05})


def _contrastive(model, k, alpha, max_length=4):
    src_input = torch.zeros(1, 3, dtype=torch.long)
    with torch.no_grad():
        return model.generate(src_input, max_length=max_length, decoding_strategy='contrastive', k=k, alpha=alpha,
                              SOS_token=SOS, EOS_token=EOS).tolist()


@pytest.mark.order(37)
def test_contrastive_decoding():
    # With one-hot hidden states the degeneration penalty of a candidate is 1
    # if it already appears in the sequence (SOS included) and 0 otherwise,
    # so score(v) = (1 - alpha) * P(v) - alpha * [v already generated].
    model = _bigram_model(_CONTRASTIVE_PROBS)

    # alpha = 0.6, k = 3: 1 -> 2 (1 is penalized) -> 3 -> 1 (all penalized, highest P wins)
    assert _contrastive(model, k=3, alpha=0.6) == [[1, 2, 3, 1]], \
        "Contrastive decoding did not produce expected tokens."
    # k = 2 restricts the candidates to {1, 2}
    assert _contrastive(model, k=2, alpha=0.6) == [[1, 2, 1, 1]], \
        "Contrastive decoding must only consider the top-k candidates."
    # alpha = 0.3: model confidence dominates and repeating 1 is still the best option
    assert _contrastive(model, k=3, alpha=0.3) == [[1, 1, 1, 1]], \
        "Contrastive decoding does not weight confidence by (1 - alpha) and the penalty by alpha."
    # alpha = 0 is plain greedy decoding
    assert _contrastive(model, k=3, alpha=0.0) == [[1, 1, 1, 1]]
