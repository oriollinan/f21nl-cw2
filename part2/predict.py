import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch

from model import GPT, GPTConfig
from vocab import Vocab


def visualize_attention(
    attention_scores: list[list[torch.Tensor]],
    source_tokens: list[str],
    target_tokens: list[str],
    output_path="attention_heatmap.png",
    cmap: str = "YlGnBu",
) -> None:
    """Visualize the attention scores as a heatmap.

    Args:
        attention_scores (list[list[torch.Tensor]]): the attention scores for each time step and
            each for each layer. Each element in the outer list corresponds to a time step, and
            each element in the inner list corresponds to a layer. Each tensor has shape
            (bsz, num_heads, seq_len, seq_len) where seq_len is the length of the input sequence
            at that time step.
        source_tokens (list[str]): list of source tokens
        target_tokens (list[str]): list of target tokens
        output_path (str, optional): path to save the heatmap. Defaults to "attention_heatmap.png".
        cmap (str, optional): colormap to use. Defaults to "YlGnBu".
    """
    # We only visualize the attention scores from the last time step
    attention_scores = attention_scores[-1]
    num_layers = len(attention_scores)
    num_heads = attention_scores[0].size(1)

    # Create a heatmap plot with subplots for each layer and head
    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(num_heads * 8, num_layers * 8)
    )
    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer, head] if num_layers > 1 else axes[head]
            # This is of shape (seq_len, seq_len)
            attn = attention_scores[layer][0, head].cpu().numpy()

            ax.imshow(attn, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(source_tokens) + len(target_tokens) - 1))
            ax.set_xticklabels(source_tokens + target_tokens[:-1], rotation=90)
            ax.set_yticks(np.arange(len(source_tokens) + len(target_tokens) - 1))
            ax.set_yticklabels(source_tokens + target_tokens[:-1])
            ax.set_title(
                f"Layer {layer + 1} Head {head + 1}", fontsize=32, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_path)


def predict(args):
    print(f"load vocab from {args.vocab_file}")
    vocab = Vocab.load(args.vocab_file)

    print(f"Load model from {args.checkpoint_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = None
    if args.checkpoint_path:
        model = GPT.load(args.model_config, args.checkpoint_path, device)
    else:
        model = GPT.load_wandb(args.model_config, args.wandb_checkpoint_run, device)

    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load("vocab/spm.model")
    with torch.inference_mode():
        subword_tokens = sp.encode_as_pieces(args.sentence)
        inputs = vocab.src.to_input_tensor([subword_tokens], device=device)

        model_output = model.generate(
            input_ids=inputs,
            max_decoding_time_steps=args.max_decoding_time_step,
            output_attentions=True,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )

        generated_output, attention_scores = (
            model_output.sequences[0],
            model_output.attentions,
        )

        predictions = (
            "".join(
                [
                    vocab.src.id2word[y_t.item()]
                    for y_t in generated_output[len(inputs[0]) :]
                ]
            )
            .replace("▁", " ")
            .strip()
        )

        print(f"Prediction: {predictions}")
        visualize_attention(
            attention_scores,
            source_tokens=[vocab.src.id2word[y_t.item()] for y_t in inputs[0]],
            target_tokens=[
                vocab.src.id2word[y_t.item()]
                for y_t in generated_output[len(inputs[0]) :]
            ],
            output_path="attention_heatmap.png",
            cmap="YlGnBu",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sentence",
        type=str,
        required=True,
        help="Input sentence to translate.",
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Path to the model checkpoint.",
    )

    group.add_argument(
        "--wandb-checkpoint-run",
        type=str,
        help="Path to the model checkpoint in wandb.",
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to the model YAML config.",
    )

    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=Path("vocab", "vocab.json"),
        help="Path to the vocabulary file.",
    )

    parser.add_argument(
        "--max-decoding-time-step",
        type=int,
        default=70,
        help="Maximum number of time steps to unroll during decoding.",
    )

    # ONLY required for MSc students
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to use sampling, use greedy decoding otherwise.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="If specified, only the top k tokens with the highest probability are considered for generation.",
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        help="If specified, only the smallest set of tokens with cumulative probability >= top_p are considered for generation.",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to module the next token probabilities.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Set random seed for reproducibility
    # DO NOT CHANGE!
    torch.manual_seed(42)
    predict(args)
