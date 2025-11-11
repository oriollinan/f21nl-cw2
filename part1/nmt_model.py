import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils


class NMT(nn.Module):
    """Simple Neural Machine Translation Model"""

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_token_idx: int = 0,
        tgt_pad_token_idx: int = 0,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize the NMT Model.
        Args:
            embed_size (int): Embedding size (dimensionality) for the words.
            hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            src_pad_token_idx (int): Padding token index for source language.
            tgt_pad_token_idx (int): Padding token index for target language.
            dropout_rate (float): Dropout probability, for attention
        """
        super().__init__()

        self.source_embeddings = torch.nn.Embedding(
            src_vocab_size, embed_size, padding_idx=src_pad_token_idx
        )
        self.target_embeddings = torch.nn.Embedding(
            tgt_vocab_size, embed_size, padding_idx=tgt_pad_token_idx
        )

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_token_idx = src_pad_token_idx
        self.tgt_pad_token_idx = tgt_pad_token_idx
        self.dropout_rate = dropout_rate

        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            bidirectional=True,
        )
        self.decoder = nn.LSTMCell(
            input_size=embed_size + hidden_size, hidden_size=hidden_size, bias=True
        )
        self.h_projection = nn.Linear(
            in_features=hidden_size * 2, out_features=hidden_size, bias=False
        )
        self.c_projection = nn.Linear(
            in_features=hidden_size * 2, out_features=hidden_size, bias=False
        )
        self.att_projection = nn.Linear(
            in_features=hidden_size * 2, out_features=hidden_size, bias=False
        )
        self.combined_output_projection = nn.Linear(
            in_features=hidden_size * 3, out_features=hidden_size, bias=False
        )
        self.target_vocab_projection = nn.Linear(
            in_features=hidden_size, out_features=tgt_vocab_size, bias=False
        )
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass and compute the log-likelihood of target sentences.

        Args:
            source (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                b = batch_size, src_len = maximum source sentence length.
            target (Tensor): Tensor of padded target sentences with shape (tgt_len, b), where
                b = batch_size, tgt_len = maximum target sentence length.

        Returns:
            scores (Tensor): a variable/tensor of shape (b, ) representing the log-likelihood
                of generating the gold-standard target sentence for each example in the input
                batch. Here b = batch size.

        Returns:
            scores (Tensor): a variable/tensor of shape (b, ) representing the log-likelihood of
                generating the gold-standard target sentence for each example in the input batch.
                Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = (source != self.src_pad_token_idx).sum(dim=0).tolist()

        # Run the network forward:
        # 1. Apply the encoder to `source_padded` by calling `self.encode()`
        # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        # 3. Apply the decoder to compute combined-output by calling `self.decode()`
        # 4. Compute log probability distribution over the target vocabulary
        enc_hiddens, dec_init_state = self.encode(source, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target != self.tgt_pad_token_idx).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = (
            torch.gather(P, index=target[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        )
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(
        self, source_padded: torch.Tensor, source_lengths: list[int]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply the encoder to source sentences to obtain encoder hidden states.
            Also projects the final state of the encoder to obtain initial states for decoder.

        Args:
            source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                b = batch_size, src_len = maximum source sentence length. Note that
               these have already been sorted in order of longest to shortest sentence.

            source_lengths (list[int]): list of actual lengths for each of the source sentences in the batch

        Returns:
            enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                b = batch size, src_len = maximum source sentence length, h = hidden size.

            dec_init_state (tuple(Tensor, Tensor)): tuple of tensors representing the decoder's initial
                hidden state and cell.
        """
        # 1. Construct Tensor `X` of source sentences with shape (src_len, b, e)
        X = self.source_embeddings(source_padded)

        # 2. Compute enc_hiddens, last_hidden, last_cell
        # Pack the padded sequence before passing to encoder
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            X, source_lengths, enforce_sorted=False
        )
        enc_hiddens_packed, (last_hidden, last_cell) = self.encoder(X_packed)
        # Pad the packed sequence
        enc_hiddens, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_hiddens_packed)
        # Change shape from (src_len, b, h*2) to (b, src_len, h*2)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)

        # 3. Compute dec_init_state
        # last_hidden and last_cell are shape (2, b, h) - concatenate forwards and backwards
        # Concatenate along the last dimension to get (b, 2*h)
        init_decoder_hidden = self.h_projection(
            torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        )
        init_decoder_cell = self.c_projection(
            torch.cat((last_cell[0], last_cell[1]), dim=1)
        )

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(
        self,
        enc_hiddens: torch.Tensor,
        enc_masks: torch.Tensor,
        dec_init_state: tuple[torch.Tensor, torch.Tensor],
        target_padded: torch.Tensor,
    ) -> torch.Tensor:
        """Decoding, compute the combined output vectors for a batch.

        Args:
            enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                b = batch size, src_len = maximum source sentence length, h = hidden size.
            enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                b = batch size, src_len = maximum source sentence length.
            dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
            target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                tgt_len = maximum target sentence length, b = batch size.

        Returns:
            combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop off the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        # 1. Apply attention projection layer to enc_hiddens
        enc_hiddens_proj = self.att_projection(enc_hiddens)

        # 2. Construct tensor Y of target sentences
        Y = self.target_embeddings(target_padded)

        # 3. Iterate over time dimension of Y
        for Y_t in torch.split(Y, 1, dim=0):
            # Squeeze Y_t from (1, b, e) to (b, e)
            Y_t = Y_t.squeeze(0)
            # Concatenate Y_t with o_prev to get Ybar_t
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            # Use step function to compute next decoder state and combined output
            dec_state, o_t, _ = self.step(
                Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks
            )
            # Append o_t to combined_outputs and update o_prev
            combined_outputs.append(o_t)
            o_prev = o_t

        # 4. Convert combined_outputs from list to tensor
        combined_outputs = torch.stack(combined_outputs, dim=0)

        return combined_outputs

    def step(
        self,
        Ybar_t: torch.Tensor,
        dec_state: tuple[torch.Tensor, torch.Tensor],
        enc_hiddens: torch.Tensor,
        enc_hiddens_proj: torch.Tensor,
        enc_masks: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute one forward step of the LSTM decoder, including the attention computation.

        Args:
            Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
            dec_state (tuple(Tensor, Tensor)): tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                    First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
            enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                        src_len = maximum source length, h = hidden size.
            enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                        where b = batch size, src_len = maximum source length, h = hidden size.
            enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                        where b = batch size, src_len is maximum source length.
        Returns:
            dec_state (tuple (Tensor, Tensor)): tuple of tensors both shape (b, h ), where b = batch size, h = hidden size.
                    First tensor is decoder's new hidden state, second tensor is decoder's new cell.
            combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
            alpha_t (Tensor): Tensor of shape (b, src_len), correspoding to the attention scores distribution (after softmax).
            Note: You will not use this outside of this function. We are simply returning this value so that we can sanity check
                    your implementation.
        """

        # 1. Apply decoder to Ybar_t and dec_state
        dec_hidden, dec_cell = self.decoder(Ybar_t, dec_state)
        dec_state = (dec_hidden, dec_cell)

        ### Your code here (~8-15 lines) ###

        # Dot-product attention
        # 2. Compute attention scores e_t
        # Need to compute batched matrix multiplication between dec_hidden and enc_hiddens_proj
        # dec_hidden has a shape of (b, h), enc_hiddens_proj is (b, src_len, h)
        # NOTE: enc_hiddens_proj is attn W @ enc_hiddens
        e_t = (dec_hidden.unsqueeze(1) @ enc_hiddens_proj.transpose(-1, -2)).squeeze(1)
        # We want to end up with a shape of (b, src_len)
        b = enc_hiddens_proj.shape[0]
        src_len = enc_hiddens_proj.shape[1]
        assert e_t.shape == (b, src_len)

        # If enc_masks is None, this step should be skipped
        if enc_masks != None:
            # Use bool() to convert ByteTensor to BoolTensor
            enc_masks_b = enc_masks.bool()
            # Use float("-inf") to represent -inf
            # Use masked_fill_ to fill in -inf at the masked positions
            e_t.masked_fill_(enc_masks_b, float("-inf"))

        # 3. Apply softmax to e_t to yield alpha_t of shape (b, src_len)
        alpha_t = F.softmax(e_t, dim=-1)

        # 4. Use batched matrix multiplication between alpha_t and enc_hiddens
        # alpha_t has a shape of (b, src_len), enc_hiddens is (b, src_len, 2h)
        # We want to end up with a shape of (b, 2h)
        attention_t = (alpha_t.unsqueeze(1) @ enc_hiddens).squeeze(1)
        two_h = enc_hiddens.shape[-1]
        assert attention_t.shape == (b, two_h)

        # 5. Concatenate dec_hidden with attention_t to compute tensor u_t
        u_t = torch.cat((dec_hidden, attention_t), dim=-1)

        # 6. Apply combined output projection layer to u_t to compute tensor v_t
        v_t = self.combined_output_projection(u_t)

        # 7. Compute tensor O_t by applying Tanh and then dropout to v_t
        o_t = self.dropout(F.tanh(v_t))

        ### End of your code ###
        return dec_state, o_t, alpha_t

    def generate_sent_masks(
        self, enc_hiddens: torch.Tensor, source_lengths: list[int]
    ) -> torch.Tensor:
        """Generate sentence masks for encoder hidden states.

        Args:
            enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                    src_len = max source length, h = hidden size.
            source_lengths (list[int]): list of actual lengths for each of the sentences in the batch.

        Returns:
            enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len), where
                src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(
            enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float
        )
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def generate(
        self,
        src_sent: torch.Tensor,
        tgt_start_token_id: int,
        tgt_end_token_id: int,
        max_decoding_time_step: int = 70,
        output_attentions: bool = False,
    ) -> list[int] | tuple[list[int], torch.Tensor]:
        """Given a single source sentence, generate a target sentence using greedy decoding.

        Args:
            src_sent (torch.Tensor): a single source sentence represented as a list of words
            tgt_start_token_id (int): the start-of-sentence token id in the target vocabulary
            tgt_end_token_id (int): the end-of-sentence token id in the target vocabulary
            max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
            output_attentions (bool): whether to return attention scores

        Returns:
            decoded_word_ids (list[int]): list of words in the decoded target sentence
        """
        src_encodings, dec_init_vec = self.encode(src_sent, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        y_t = torch.tensor([tgt_start_token_id], dtype=torch.long, device=self.device)

        decoded_word_ids = []
        attention_scores = []
        for t in range(max_decoding_time_step):
            y_t_embed = self.target_embeddings(y_t)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, scores = self.step(
                x,
                h_tm1,
                src_encodings,
                src_encodings_att_linear,
                enc_masks=None,
            )

            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
            if output_attentions:
                attention_scores.append(scores)

            y_t = torch.argmax(log_p_t, dim=-1)

            if y_t.item() == tgt_end_token_id:
                break
            else:
                decoded_word_ids.append(y_t.item())

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        if output_attentions:
            return decoded_word_ids, torch.stack(attention_scores).squeeze(1)
        return decoded_word_ids

    @property
    def device(self) -> torch.device:
        """Determine which device to place the Tensors upon, CPU or GPU."""
        return self.source_embeddings.weight.device

    @staticmethod
    def load(model_path: str) -> "NMT":
        """Load the model from a file.

        Args:
            model_path (str): path to the input file.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = NMT(**params["args"])
        model.load_state_dict(params["state_dict"])

        return model

    def save(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path (str): path to the output file.
        """
        print(f"save model parameters to [{path}]")

        params = {
            "args": dict(
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_pad_token_idx=self.src_pad_token_idx,
                tgt_pad_token_idx=self.tgt_pad_token_idx,
                dropout_rate=self.dropout_rate,
            ),
            "state_dict": self.state_dict(),
        }

        torch.save(params, path)
