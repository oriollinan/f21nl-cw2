import argparse
import json
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
from tqdm import tqdm

from model import GPT, GPTConfig
from vocab import Vocab


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


def evaluate(args):
    print(f"Load test source sentences from {args.test_source_file}")
    test_data_src = []
    with open(args.test_source_file, "r", encoding="utf8") as src_file:
        for line in src_file:
            test_data_src.append(line.strip())

    test_data_tgt = []
    if args.test_target_file:
        print(f"Load test target sentences from {args.test_target_file}")
        with open(args.test_target_file, "r", encoding="utf8") as tgt_file:
            for line in tgt_file:
                test_data_tgt.append(line.strip())

    print(f"Load vocab from {args.vocab_file}")
    vocab = Vocab.load(args.vocab_file)

    print(f"Loading model")

    model: GPT = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    if args.checkpoint_path != None:
        model = GPT.load(args.model_config, args.checkpoint_path, device)
    else:
        model = GPT.load_wandb(args.model_config, args.wandb_checkpoint_run, device)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.encoder))

    references = []
    predictions = []
    with torch.inference_mode():
        for src_sent in tqdm(test_data_src, desc="Generating Predictions"):
            subword_tokens = sp.encode_as_pieces(src_sent)
            inputs = vocab.src.to_input_tensor([subword_tokens], device=device)

            generated_output = model.generate(
                input_ids=inputs, 
                max_decoding_time_steps=args.max_decoding_time_step,
                output_attentions=False,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            ).sequences[0]

            predictions.append(
                "".join(
                    [
                        vocab.src.id2word[y_t.item()]
                        for y_t in generated_output[len(inputs[0]) : -1]
                    ]
                )
                .replace("▁", " ")
                .strip()
            )

            references.append(src_sent)

    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    bleu_score = compute_corpus_level_bleu_score(test_data_tgt, predictions)
    print(f"Corpus BLEU score = {bleu_score:.2f}")

    print(f"Saving predictions to {output_file}")
    with open(output_file, "w") as f:
        data = {
            "bleu_score": bleu_score,
            "predictions": predictions,
            "references": test_data_tgt,
        }
        json.dump(data, f, indent=4)
    return bleu_score


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
        "--model-config",
        type=Path,
        required=True,
        help="Path to the model YAML config.",
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
        "--encoder",
        type=Path,
        default=Path("vocab/spm.model"),
        help="Path to the encoder model.",
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

    evaluate(args)
