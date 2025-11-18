import argparse
import json
from pathlib import Path

import sacrebleu
import torch
from tqdm import tqdm

from nmt_model import NMT
from utils import read_corpus
from vocab import Vocab

import wandb

def compute_corpus_level_bleu_score(
    references: list[list[str]], predictions: list[list[str]]
) -> float:
    """Compute corpus-level BLEU score, given decoding results and reference sentences.

    Args:
        references (list[list[str]]): a list of gold-standard reference target sentences
        predictions (list[list[str]]): a list of predictions, one for each reference

    Returns:
        bleu_score (float): corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == "<s>":
        references = [ref[1:-1] for ref in references]

    # detokenize the subword pieces to get full sentences
    detokened_refs = [
        "".join(pieces).replace("▁", " ").strip() for pieces in references
    ]
    detokened_preds = [
        "".join(value).replace("▁", " ").strip() for value in predictions
    ]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_preds, [detokened_refs])

    return bleu.score


def evaluate(args: argparse.Namespace):
    """Performs decoding on a test set, and save the results."""

    print(f"Load test source sentences from {args.test_source_file}")
    test_data_src = read_corpus(args.test_source_file, source="src")

    test_data_tgt = []
    if args.test_target_file:
        print(f"Load test target sentences from {args.test_target_file}")
        test_data_tgt = read_corpus(args.test_target_file, source="tgt")

    print(f"Load vocab from {args.vocab_file}")
    vocab = Vocab.load(args.vocab_file)

    print(f"Loading model")
    model = NMT.load(args.checkpoint_path) if args.checkpoint_path != None else NMT.load_wandb(args.wandb_checkpoint_run)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.inference_mode():
        for src_sent in tqdm(test_data_src, desc="Generating Predictions"):
            src_sents_var = vocab.src.to_input_tensor([src_sent], model.device)

            generated_output = model.generate(
                src_sents_var,
                tgt_start_token_id=vocab.tgt["<s>"],
                tgt_end_token_id=vocab.tgt["</s>"],
                max_decoding_time_step=args.max_decoding_time_step,
            )

            predictions.append([vocab.tgt.id2word[y_t] for y_t in generated_output])

    bleu_score = None
    if test_data_tgt:
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, predictions)
        print(f"Corpus BLEU score = {bleu_score:.2f}")

    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving predictions to {output_file}")
    with open(output_file, "w") as f:
        if test_data_tgt[0][0] == "<s>":
            test_data_tgt = [ref[1:-1] for ref in test_data_tgt]

        data = {
            "bleu_score": bleu_score,
            "predictions": [
                "".join(value).replace("▁", " ").strip() for value in predictions
            ],
            "references": [
                "".join(pieces).replace("▁", " ").strip() for pieces in test_data_tgt
            ],
        }
        json.dump(data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--test-source-file",
        type=Path,
        default=Path("..", "multi30k_data", "test_2016_flickr.en"),
        help="Path to the test source file, containing source sentences.",
    )

    parser.add_argument(
        "--test-target-file",
        type=Path,
        default=Path("..", "multi30k_data", "test_2016_flickr.fr"),
        help="Path to the test target file, containing gold-standard target sentences.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs", "results.json"),
        help="Path to the output file, where the best-scoring decoding results will be saved.",
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

    evaluate(args)
