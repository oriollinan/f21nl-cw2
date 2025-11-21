import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm
import torch

from nmt_model import NMT
from vocab import Vocab


def visualize_attention(
    attention_scores: torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    output_path="attention_heatmap.png",
    cmap: str = "YlGnBu",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        attention_scores.cpu().numpy(),
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap=cmap,
        ax=ax,
    )

    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title("Attention Heatmap")

    plt.savefig(output_path)


def predict(args):
    print(f"load vocab from {args.vocab_file}")
    vocab = Vocab.load(args.vocab_file)

    print(f"Load model from {args.checkpoint_path}")
    model = (
        NMT.load(args.checkpoint_path)
        if args.checkpoint_path
        else NMT.load_wandb(args.wandb_checkpoint_run)
    )
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load("src.model")
    with torch.inference_mode():
        subword_tokens = sp.encode_as_pieces(args.sentence)
        src_sents_var = vocab.src.to_input_tensor([subword_tokens], model.device)
        generated_output, attention_scores = model.generate(
            src_sents_var,
            tgt_start_token_id=vocab.tgt["<s>"],
            tgt_end_token_id=vocab.tgt["</s>"],
            max_decoding_time_step=args.max_decoding_time_step,
            output_attentions=True,
        )

        predictions = (
            "".join([vocab.tgt.id2word[y_t] for y_t in generated_output])
            .replace("▁", " ")
            .strip()
        )

        print(f"Prediction: {predictions}")
        visualize_attention(
            attention_scores,
            source_tokens=subword_tokens,
            target_tokens=[vocab.tgt.id2word[y_t] for y_t in generated_output]
            + ["</s>"],
            output_path="attention_heatmap.png",
            cmap="viridis",
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    predict(args)
