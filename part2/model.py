import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.nn import functional as F

import wandb


@dataclass
class GPTAttentionOutput:
    """Output of the GPT attention layer."""

    output: torch.Tensor
    attentions: torch.Tensor | None = None


@dataclass
class GPTBlockOutput:
    """Output of the GPT block."""

    output: torch.Tensor
    attentions: torch.Tensor | None = None


@dataclass
class GPTForwardOutput:
    """Output of the GPT model."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    attentions: list[torch.Tensor] | None = None


@dataclass
class GPTGenerateOutput:
    """Output of the GPT generate method."""

    sequences: torch.Tensor
    attentions: list[list[torch.Tensor]] | None = None


@dataclass
class GPTConfig:
    """GPT config.

    The parameters are initialized to GPT-2 values.
    """

    # GPT-2 like block size
    block_size: int = 1024
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304

    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

    dropout: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "GPTConfig":
        """Create a GPTConfig from a yaml file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_dict(self) -> dict:
        """Convert the GPTConfig to a dictionary."""
        return {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embed": self.n_embed,
            "dropout": self.dropout,
        }


class CausalSelfAttention(nn.Module):
    """Vanilla multi-head masked self-attention layer with a projection at the end."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)

        # Regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Output projection
        self.proj = nn.Linear(config.n_embed, config.n_embed)

        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> GPTAttentionOutput:
        """Forward the masked self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where B is the batch size,
                T is the sequence length, C is the embedding dimension.
            output_attentions (bool, optional): Whether to return the attention scores. Defaults to False.

        Returns:
            GPTAttentionOutput: A dataclass with the following fields:
                output: Tensor of shape (B, T, C) where B is the batch size,
                    T is the sequence length, C is the embedding dimension.
                attentions: (optional) Tensor of shape (B, nh, T, T) where nh is the number of heads.
        """
        B, T, C = x.size()
        ### Your code here (~8-15 lines) ###
        # Step 1: Calculate query, key, values for all heads
        # (B, nh, T, hs)
        hs = int(C / self.n_head)
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)
        assert q.shape == (B, self.n_head, T, hs)
        assert k.shape == (B, self.n_head, T, hs)
        assert v.shape == (B, self.n_head, T, hs)

        # Step 2: Compute attention scores
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attention = q @ k.transpose(-2, -1)
        attention /= hs**0.5
        assert attention.shape == (B, self.n_head, T, T)

        # Step 3: Masking out the future tokens (causal) and softmax
        mask = self.mask[:, :, :T, :T]
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)

        # Step 4: Compute the attention output
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = attention @ v
        assert y.shape == (B, self.n_head, T, hs)

        # Step 5: re-assemble all head outputs side by side
        y = y.transpose(1, 2)
        # (B, T, nh, hs) -> (B, T, C)
        # NOTE: can't use `view` since `y` is not in a contiguos block of memory
        y = y.reshape(B, T, C)
        assert y.shape == (B, T, C)

        # Step 6: output projection + dropout
        y = self.proj(y)
        y = self.resid_drop(y)

        ### End of your code ###
        return GPTAttentionOutput(
            output=y, attentions=attention if output_attentions else None
        )


class Block(nn.Module):
    """A transformer block: communication followed by computation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> GPTBlockOutput:
        """Forward the GPT block."""
        # Step 1: communication
        # Layer norm -> self-attention -> residual connection
        pre_layer = self.ln1(x)

        attention_output = self.attn(pre_layer, output_attentions=output_attentions)

        x = x + attention_output.output

        # Step 2: computation
        # Layer norm -> MLP -> residual connection
        x = x + self.mlp(self.ln2(x))

        return GPTBlockOutput(output=x, attentions=attention_output.attentions)


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding table
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # Decoder head
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.drop = nn.Dropout(config.dropout)

        self.block_size = config.block_size

        self.apply(self._init_weights)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params} ({total_params / 1e6:.2f}M)")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> GPTForwardOutput:
        """Forward the GPT model.

        Args:
            input_ids (torch.Tensor): Tensor of shape (b, t) where b is the batch size,
                t is the sequence length.
            targets (torch.Tensor, optional): Tensor of shape (b, t) where b is the batch size,
                t is the sequence length. Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor of shape (b, t, vocab_size) where b is the batch size,
                t is the sequence length, vocab_size is the size of the vocabulary.
            loss (torch.Tensor, optional): Cross-entropy loss if targets is provided.
        """
        attention_scores = []
        b, t = input_ids.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward, model block size ({t}, {self.block_size}) is exhausted."

        token_embeddings = self.tok_emb(input_ids)

        # Positional embeddings: each position maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]
        x_input = token_embeddings + position_embeddings

        x = self.drop(x_input)
        # We use a loop here instead of nn.Sequential so that we can
        # collect the attention scores from each layer
        for block in self.blocks:
            block_output = block(x, output_attentions=output_attentions)

            x = block_output.output
            attention_scores.append(block_output.attentions)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # Use ignore_index=0 to ignore padding tokens in the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0
            )

        return GPTForwardOutput(logits=logits, loss=loss, attentions=attention_scores)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_decoding_time_steps: int = 20,
        output_attentions: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int = 2,
    ) -> GPTGenerateOutput:
        """Sample from the model.

        Take a conditioning sequence of indices input_ids (LongTensor of shape (b,t)) and complete
        the sequence max_decoding_time_steps times, feeding the predictions back into the model each time.

        Args:
            input_ids (torch.Tensor): Tensor of shape (b, t) where b is the batch size,
                t is the sequence length.
            max_decoding_time_steps (int): Number of tokens to generate.
            output_attentions (bool, optional): Whether to return the attentions. Defaults to False.
            do_sample (bool, optional): Whether to use sampling or greedy decoding.
                Defaults to False (greedy decoding).
            temperature (float, optional): The value used to module the next token probabilities.
                Defaults to 1.0.
            top_k (int, optional): If specified, only the top k tokens with the highest
                probability are considered for generation. Defaults to None.
            top_p (float, optional): If specified, only the smallest set of tokens with
                cumulative probability >= top_p are considered for generation. Defaults to None.
        Returns:
            GPTGenerateOutput: A dataclass with the following fields:
                sequences: Tensor of shape (b, t+max_new_tokens) with the generated sequences.
                attentions: (optional) List of attention scores from each layer.
        """
        attentions = [] if output_attentions else None
        for _ in range(max_decoding_time_steps):
            # if the sequence context is growing too long we must crop it at block_size
            current_input_ids = (
                input_ids
                if input_ids.size(1) <= self.config.block_size
                else input_ids[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            output = self(current_input_ids, output_attentions=output_attentions)

            # pluck the logits at the final step and scale by desired temperature
            logits = output.logits
            logits = logits[:, -1, :] / temperature

            if do_sample:
                ### Your code here (~5-12 lines) ###
                # 1. If top_k is not None, crop the logits to only the top k options
                top_k_indices = torch.arange(logits.shape[-1], device=logits.device)
                if top_k is not None:
                    logits, top_k_indices = torch.topk(logits, top_k)

                # 2. If top_p is not None, crop the logits to only the top p options
                top_p_indices = torch.arange(logits.shape[-1], device=logits.device)
                if top_p is not None:
                    logits, top_p_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(logits, -1)
                    cum_probs = torch.cumsum(probs, -1)
                    # NOTE: Find the index where top_p becomes greater than cum probs
                    cutoff = torch.searchsorted(cum_probs, top_p)
                    cutoff = int(cutoff.clamp(min=0, max=logits.shape[0] - 1))
                    # NOTE: cutoff is off by one
                    logits = logits[: cutoff + 1]
                    top_p_indices = top_p_indices[: cutoff + 1]

                # apply softmax to convert logits to (normalized) probabilities
                logits = F.softmax(logits, -1)

                # sample from the distribution using the re-normalized probabilities
                sample = torch.multinomial(logits, 1)
                predicted_id = top_k_indices[top_p_indices[sample]]

                # append sampled index to the running sequence and continue
                input_ids = torch.cat((input_ids, predicted_id), dim=1)
                ### End of your code ###
            else:
                # greedily take the argmax
                predicted_id = torch.argmax(logits, dim=-1, keepdim=True)
                # append predicted index to the running sequence and continue
                input_ids = torch.cat((input_ids, predicted_id), dim=1)

            if output.attentions is not None and attentions is not None:
                attentions.append(output.attentions)

            if predicted_id == eos_token_id:
                break

        return GPTGenerateOutput(sequences=input_ids, attentions=attentions)

    @staticmethod
    def load(model_config, checkpoint_path: str, device):
        model = GPT(GPTConfig.from_yaml(model_config))
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model.to(device)

    @staticmethod
    def load_wandb(model_config, run_path: str, device):
        filename = "models/gpt.model"

        wandb.restore(name=filename, run_path=run_path, replace=True)

        return GPT.load(model_config, filename, device)

    def save(self, path: str):
        full_path = str(Path(path, "gpt.model"))
        torch.save(
            self.state_dict(),
            full_path,
        )

        wandb.save(full_path)
