import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

try:
    from decoder import TransformerDecoder
    from encoder import TransformerEncoder
except ModuleNotFoundError:
    from src.decoder import TransformerDecoder
    from src.encoder import TransformerEncoder

class Transformer(nn.Module):
    """Transformer model.

    This class implements the full Transformer model, consisting of an encoder and a decoder.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        tgt_vocab_size (int): The size of the target vocabulary.
        max_enc_position_embeddings (int): The maximum number of positions for positional embeddings (encoder side).
        max_dec_position_embeddings (int): The maximum number of positions for positional embeddings (decoder side).
        enc_d_model (int): The dimension of the input embeddings (encoder side).
        dec_d_model (int): The dimension of the input embeddings (decoder side).
        enc_num_attention_heads (int): The number of attention heads in the multi-head attention mechanisms (encoder side).
        dec_num_attention_heads (int): The number of attention heads in the multi-head attention mechanisms (decoder side).
        enc_intermediate_size (int): The dimension of the feed-forward network's intermediate layer (encoder side).
        dec_intermediate_size (int): The dimension of the feed-forward network's intermediate layer (decoder side).
        num_enc_hidden_layers (int): The number of Transformer encoder layers to stack.
        num_dec_hidden_layers (int): The number of Transformer decoder layers to stack.

    Attributes:
        encoder (TransformerEncoder): Transformer encoder.
        decoder (TransformerDecoder): Transformer decoder.
        output_linear (nn.Linear): Linear layer to project the decoder output to the target vocabulary size.
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, max_enc_position_embeddings: int, max_dec_position_embeddings: int,
                enc_d_model: int, dec_d_model: int, num_attention_heads: int, enc_intermediate_size: int, 
                dec_intermediate_size: int, num_enc_hidden_layers: int, num_dec_hidden_layers: int
                ):
        
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            max_position_embeddings=max_enc_position_embeddings,
            d_model=enc_d_model,
            num_attention_heads=num_attention_heads,
            intermediate_size=enc_intermediate_size,
            num_hidden_layers=num_enc_hidden_layers
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            max_position_embeddings=max_dec_position_embeddings,
            d_model=dec_d_model,
            num_attention_heads=num_attention_heads,
            intermediate_size=dec_intermediate_size,
            num_hidden_layers=num_dec_hidden_layers
        )

        self.output_linear = nn.Linear(in_features=dec_d_model, out_features=tgt_vocab_size)

    def forward(self, src_input: torch.Tensor, tgt_input: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the Transformer model.

        Args:
            src_input (torch.Tensor): Input tensor of shape (batch_size, src_seq_len).
            tgt_input (torch.Tensor): Target tensor of shape (batch_size, tgt_seq_len).
            attn_mask (torch.Tensor): Attention mask tensor for encoder tensor due to padding (batch_size, src_seq_len, src_seq_len). Default to None

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        # Pass the source input through the encoder
        enc_output = self.encoder(src_input, attn_mask)

        # Pass the target input through the decoder, with the encoder output
        dec_output = self.decoder(tgt_input, enc_output)

        # Project the decoder output to the target vocabulary size
        dec_output = self.output_linear(dec_output)

        return dec_output
    
    def generate(self, src_input: torch.Tensor, max_length: int = 50, decoding_strategy: str = 'greedy', **kwargs) -> torch.Tensor:
        """Generate a sequence given a source input using different decoding strategies.

        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            max_length (int, optional): Maximum length of the generated sequence. Defaults to 50.
            decoding_strategy (str, optional): Decoding strategy ('greedy', 'beam_search', 'top_k', 'top_p').
            **kwargs: Additional arguments specific to the decoding strategy.

        Returns:
            torch.Tensor: Generated sequence of token IDs of shape (batch_size, generated_seq_len).
        """
        if decoding_strategy == 'greedy':
            return self.__greedy_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'beam_search':
            return self.__beam_search_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'sampling':
            return self.__sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'top_k':
            return self.__top_k_sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'top_p':
            return self.__top_p_sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'contrastive':
            return self.__contrastive_decode(src_input, max_length, **kwargs)
        
        else:
            raise ValueError(f"Invalid decoding strategy: {decoding_strategy}")

    def __greedy_decode(self, src_input: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        """Generate a sequence using greedy decoding.

        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            max_length (int): Maximum length of the generated sequence.
            **kwargs: Additional arguments (e.g., start token, end token, device, ...).

        Returns:
            torch.Tensor: Generated sequence of token IDs of shape (batch_size, generated_seq_len).
        """
        # Pass the source input through the encoder
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        # Get start and end tokens
        SOS_token = kwargs.get('SOS_token', 2)  # Default SOS token index
        EOS_token = kwargs.get('EOS_token', 3)  # Default EOS token index

        # Initialize the target sequence with SOS_token
        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            # Pass through the decoder
            dec_output = self.decoder(tgt_input, enc_output)
            # Project the decoder output to vocabulary size
            dec_output = self.output_linear(dec_output)
            # Get the logits for the last time step
            logits = dec_output[:, -1, :] # Shape: (batch_size, vocab_size)
            # Get the token with the highest probability
            next_token = torch.argmax(logits, dim=1, keepdim=True)  # Shape: (batch_size, 1)
            # Append the next token to the target sequence
            tgt_input = torch.cat((tgt_input, next_token), dim=1)
            # Check if all sequences have generated EOS_token
            if (next_token == EOS_token).all():
                break

        # Return the generated sequences (excluding the first SOS token)
        generated_sequence = tgt_input[:, 1:] # Shape: (batch_size, seq_len)
        return generated_sequence

    def __beam_search_decode(self, src_input: torch.Tensor, max_length: int, beam_size: int = 3, **kwargs) -> torch.Tensor:
        """Generate a sequence using beam search decoding.

        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            max_length (int): Maximum length of the generated sequence.
            beam_size (int, optional): Beam size for beam search. Defaults to 3.
            **kwargs: Additional arguments (e.g., start token, end token, device, ...).

        Returns:
            torch.Tensor: Generated sequence of token IDs of shape (batch_size, generated_seq_len).
        """
        # Note: For simplicity, batch_size = 1 is assumed
        batch_size = src_input.size(0)
        if batch_size != 1:
            raise NotImplementedError("Beam search decoding currently only supports batch_size=1")
        device = src_input.device

        # Pass the source input through the encoder
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        # Get start and end tokens
        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        # Initialize the beam with the start token
        tgt_input = torch.full(size=(batch_size, 1), fill_value=SOS_token, dtype=torch.long, device=device)
        beam = [(tgt_input, 0)]  # Each item is (sequence tensor, cumulative log probability)

        for _ in range(max_length):
            candidates = []
            for seq, score in beam:
                if seq[0, -1].item() == EOS_token:
                    # If EOS token is reached, add the sequence to candidates without expanding
                    candidates.append((seq, score))
                    continue

                # Pass through the decoder
                dec_output = self.decoder(seq, enc_output)
                # Project to vocabulary size
                dec_output = self.output_linear(dec_output)
                # Get the logits for the last time step
                logits = dec_output[:, -1, :]  # Shape: (1, vocab_size)
                # Apply log softmax to get log probabilities
                log_probs = torch.log_softmax(logits, dim=1)  # Shape: (1, vocab_size)
                for next_token in range(log_probs.size(1)):
                    next_token_tensor = torch.full(size=(batch_size, 1), fill_value=next_token, dtype=torch.long, device=device)
                    new_seq = torch.cat((seq, next_token_tensor), dim=1) # Shape: (1, seq_len+1)
                    new_score = log_probs[:, next_token].item()
                    candidates.append((new_seq, new_score))
            # Select top beam_size sequences
            beam = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]
            # If all sequences have reached EOS, stop
            if all(seq[0, -1].item() == EOS_token for seq, _ in beam):
                break
        # Return the sequence with the highest score
        best_seq = max(beam, key=lambda tup: tup[1])[0]
        # Remove the SOS token
        generated_sequence = best_seq[:, 1:]  # Shape: (1, seq_len)
        return generated_sequence
    
    def __sampling_decode(self, src_input: torch.Tensor, max_length: int, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        """Generate a sequence using multinomial sampling with temperature.

        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            max_length (int): Maximum length of the generated sequence.
            temperature (float, optional): Temperature parameter to adjust the sharpness of the probability distribution. Defaults to 1.0.
            **kwargs: Additional arguments (e.g., start token, end token, device, ...).

        Returns:
            torch.Tensor: Generated sequence of token IDs of shape (batch_size, generated_seq_len).
        """
        # Pass the source input through the encoder
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        # Get start and end tokens
        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        # Initialize the target sequence with SOS_token
        tgt_input = torch.full(size=(batch_size, 1), fill_value=SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            # Pass through the decoder
            dec_output = self.decoder(tgt_input, enc_output)
            # Project to vocabulary size
            dec_output = self.output_linear(dec_output)
            # Get the logits for the last time step
            logits = dec_output[:, -1, :]  # Shape: (batch_size, vocab_size)

            # Apply temperature scaling to the logits
            scaled_logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(scaled_logits, dim=1)

            # Sample from the probability distribution
            next_token = torch.multinomial(probs, 1)  # Shape: (batch_size, 1)

            # Append the next token to tgt_input
            tgt_input = torch.cat((tgt_input, next_token), dim=1)

            # Check if all sequences have generated EOS_token
            if (next_token == EOS_token).all():
                break

        # Return the generated sequences (excluding the first SOS token)
        generated_sequence = tgt_input[:, 1:]
        return generated_sequence

    def __top_k_sampling_decode(self, src_input: torch.Tensor, max_length: int, k: int = 10, **kwargs) -> torch.Tensor:
        """Generate a sequence using top-k sampling decoding.

        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            max_length (int): Maximum length of the generated sequence.
            k (int, optional): Number of top tokens to consider for sampling. Defaults to 10.
            **kwargs: Additional arguments (e.g., start token, end token, device, ...).

        Returns:
            torch.Tensor: Generated sequence of token IDs of shape (batch_size, generated_seq_len).
        """
        # Pass the source input through the encoder
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        # Get start and end tokens
        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        # Initialize the target sequence with SOS_token
        tgt_input = torch.full(size=(batch_size, 1), fill_value=SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            # Pass through the decoder
            dec_output = self.decoder(tgt_input, enc_output)
            # Project to vocabulary size
            dec_output = self.output_linear(dec_output)
            # Get the logits for the last time step
            logits = dec_output[:, -1, :] # Shape: (batch_size, vocab_size)
            # Apply log softmax to get log probabilities
            log_probs = torch.softmax(logits, dim=1)
            # Get the top k tokens
            topk_log_probs, topk_indices = torch.topk(log_probs, k=k, dim=1)
            # Sample from the top k tokens
            probs = topk_log_probs / topk_log_probs.sum(dim=1, keepdim=True)
            next_token = torch.multinomial(probs, 1)  # Shape: (batch_size, 1)
            # Map sampled indices to original token indices
            next_token = torch.gather(topk_indices, 1, next_token)
            # Append next token to tgt_input
            tgt_input = torch.cat((tgt_input, next_token), dim=1)
            # Check if all sequences have generated EOS_token
            if (next_token == EOS_token).all():
                break

        # Return the generated sequences (excluding the first SOS token)
        generated_sequence = tgt_input[:, 1:]
        return generated_sequence
    

    def __top_p_sampling_decode(self, src_input: torch.Tensor, max_length: int, p: float = 0.9, **kwargs) -> torch.Tensor:
        """Generate a sequence using top-p (nucleus) sampling decoding.

        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            max_length (int): Maximum length of the generated sequence.
            p (float, optional): Cumulative probability threshold. Defaults to 0.9.
            **kwargs: Additional arguments (e.g., start token, end token, device, ...).

        Returns:
            torch.Tensor: Generated sequence of token IDs of shape (batch_size, generated_seq_len).
        """
        # Pass the source input through the encoder
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = None

        batch_size = src_input.size(0)
        device = src_input.device

        # Get start and end tokens
        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        # Initialize the target sequence with SOS_token
        tgt_input = None

        for _ in range(max_length):
            # Pass through the decoder
            dec_output = None
            # Project to vocabulary size
            dec_output = None
            # Get the logits for the last time step
            logits = None  # Shape: (batch_size, vocab_size)
            # Apply softmax to get probabilities
            probs = None
            # Sort the probabilities
            sorted_probs, sorted_indices = None
            # Compute cumulative probabilities
            cumulative_probs = None
            # Keep the smallest set of top tokens whose cumulative probability exceeds p
            # (the token that crosses p is kept) and remove the rest
            sorted_indices_to_remove = None
            sorted_probs[sorted_indices_to_remove] = 0
            # Normalize the probabilities
            sorted_probs = None
            # Sample from the filtered distribution
            next_token = None
            # Map sampled indices to original token indices
            next_token = None
            # Append next token to tgt_input
            tgt_input = None
            # Check if all sequences have generated EOS_token
            if (next_token == EOS_token).all():
                break

        # Return the generated sequences (excluding the first SOS token)
        generated_sequence = None
        return generated_sequence
    
    def __contrastive_decode(self, src_input: torch.Tensor, max_length: int, k: int = 5, alpha: float = 0.6, **kwargs) -> torch.Tensor:
        """Generate a sequence using contrastive decoding (contrastive search) for batch sizes > 1.

        Args:
            src_input (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            max_length (int): Maximum length of the generated sequence.
            k (int, optional): Number of top tokens to consider. Defaults to 5.
            alpha (float, optional): Weighting factor between model confidence and degeneration penalty. Defaults to 0.6.
            **kwargs: Additional arguments (e.g., start token, end token, device, attention masks).

        Returns:
            torch.Tensor: Generated sequence of token IDs of shape (batch_size, generated_seq_len).
        """
        # Pass the source input through the encoder
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = None

        batch_size = src_input.size(0)
        device = src_input.device

        # Get start and end tokens
        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        # Initialize the target sequence with SOS_token
        tgt_input = None

        for _ in range(max_length):
            # Pass through the decoder
            dec_output = None
            # Project to vocabulary size
            dec_output = None
            # Get the logits for the last time step
            logits = None  # Shape: (batch_size, vocab_size)
            # Apply log softmax to get log probabilities
            probs = None
            # Get the top k tokens
            topk_probs, topk_indices = None

            # Prepare tensors for all candidates
            expanded_tgt_input = None  # Shape: (k, seq_len)
            next_tokens = None  # Shape: (k, 1)
            y_candidates = None  # Shape: (k, seq_len + 1)

            # Pass each candidate through the decoder
            dec_outputs_candidate = None

            # Extract hidden states
            h_v = None  # Shape: (k, hidden_size)
            h_j = None  # Shape: (k, seq_len, hidden_size)

            # Normalize hidden states
            h_v_norm = None  # Shape: (k, hidden_size)
            h_j_norm = None  # Shape: (k, seq_len, hidden_size)

            # Compute cosine similarities between h_v and each h_j
            cos_sim = None  # Shape: (k, seq_len)

            # Get maximum cosine similarity for each candidate
            max_sim = None  # Shape: (k,)

            # Compute scores
            P_LM_v = None  # Shape: (k,)
            scores = None  # Shape: (k,)

            # Select the candidate with the highest score
            best_idx = None
            best_token = None  # Shape: (1, 1)
            # Append the selected token to the target sequence
            tgt_input = None  # Shape: (1, seq_len + 1)

            # Check for EOS_token
            if best_token.item() == EOS_token:
                break

        # Return generated sequence excluding SOS_token
        generated_sequence = None
        return generated_sequence



    
if __name__ == "__main__":
    # Define parameters
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    max_position_embeddings = 128
    intermediate_size = 64

    # Define the Transformer model
    transformer = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, max_enc_position_embeddings=max_position_embeddings, 
                            max_dec_position_embeddings=max_position_embeddings, enc_d_model=d_model, dec_d_model=d_model, 
                            num_attention_heads=num_heads, enc_intermediate_size=intermediate_size, dec_intermediate_size=intermediate_size, 
                            num_enc_hidden_layers=num_layers, num_dec_hidden_layers=num_layers)

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1], attn_mask=None)
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    transformer.eval()

    # Generate random sample validation data
    val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    with torch.no_grad():

        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")
