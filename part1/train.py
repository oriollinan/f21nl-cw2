import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import wandb
from nmt_model import NMT
from utils import batch_iter, read_corpus
from vocab import Vocab


def evaluate(
    model: NMT,
    vocab: Vocab,
    dev_data: list[tuple[list[str], list[str]]],
    batch_size: int = 32,
) -> float:
    """Evaluate perplexity on dev sentences.

    Args:
        model (NMT): NMT model
        vocab (Vocab): vocabulary object
        dev_data (List[Tuple[List[str], List[str]]]): list of tuples containing source and target sentences
        batch_size (int, optional): batch size. Defaults to 32.

    Returns:
        avg_loss (float): average loss value
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.0
    cum_tgt_words = 0.0

    # inference_model() signals backend to throw away all gradients
    with torch.inference_mode():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            # Convert list of lists into tensors with padding
            # Tensor: (src_len, b)
            source_padded = vocab.src.to_input_tensor(src_sents, device=model.device)
            # Tensor: (tgt_len, b)
            target_padded = vocab.tgt.to_input_tensor(tgt_sents, device=model.device)

            loss = -model(source_padded, target_padded).sum()

            cum_loss += loss.item()
            # Omitting leading `<s>`
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict

    if was_training:
        model.train()

    return cum_loss / cum_tgt_words


def train(args: argparse.Namespace) -> None:
    """Train the NMT Model."""
    train_data_src = read_corpus(args.train_source_file, source="src")
    train_data_tgt = read_corpus(args.train_target_file, source="tgt")

    dev_data_src = read_corpus(args.dev_source_file, source="src")
    dev_data_tgt = read_corpus(args.dev_target_file, source="tgt")

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    print(f"Train dataset size: {len(train_data)}")
    print(f"Dev dataset size: {len(dev_data)}")

    print(f"Load vocab from {args.vocab_file}")
    vocab = Vocab.load(args.vocab_file)

    model_config = yaml.safe_load(args.model_config.read_text())

    model = NMT(
        embed_size=model_config["embed_size"],
        hidden_size=model_config["hidden_size"],
        dropout_rate=model_config["dropout_rate"],
        src_vocab_size=len(vocab.src),
        tgt_vocab_size=len(vocab.tgt),
        src_pad_token_idx=vocab.src["<pad>"],
        tgt_pad_token_idx=vocab.tgt["<pad>"],
    )

    model.train()

    train_config = yaml.safe_load(args.train_config.read_text())
    uniform_init = train_config["uniform_init"]
    if np.abs(uniform_init) > 0.0:
        print(
            f"uniformly initialize parameters [-{uniform_init}, +{uniform_init}]",
        )
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt["<pad>"]] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"use device: {device}")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = (
        report_tgt_words
    ) = 0
    cum_examples = report_examples = epoch = valid_num = 0
    train_time = begin_time = time.time()

    model_save_path = args.checkpoint_path.joinpath("nmt.model")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    print("Starting Training!")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**model_config, **train_config},
    )
    best_dev_loss = float("inf")
    while True:
        epoch += 1
        print(epoch)
        for src_sents, tgt_sents in batch_iter(
            train_data, batch_size=train_config["train_batch_size"], shuffle=True
        ): 
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            # Convert list of lists into tensors with padding
            # Tensor: (src_len, b)
            source_padded = vocab.src.to_input_tensor(src_sents, device=model.device)
            # Tensor: (tgt_len, b)
            target_padded = vocab.tgt.to_input_tensor(tgt_sents, device=model.device)

            example_losses = -model(source_padded, target_padded)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            # omitting leading `<s>`
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % train_config["log_every"] == 0:
                print(
                    f"epoch {epoch}, iter {train_iter}, avg. loss {report_loss / report_tgt_words:.2f} "
                    f"cum. examples {cum_examples}, speed {report_tgt_words / (time.time() - train_time):.2f} words/sec, time elapsed {time.time() - begin_time:.2f} sec"
                )

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.0
                wandb.log(
                    {
                        "iteration": train_iter,
                        "train_loss": cum_loss / cum_tgt_words,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=train_iter,
                )

            # perform validation
            if train_iter % train_config["valid_niter"] == 0:
                cum_loss = cum_examples = cum_tgt_words = 0.0
                valid_num += 1

                print("Begin validation ...")

                # compute dev perplexity and loss
                dev_loss = evaluate(
                    model=model,
                    vocab=vocab,
                    dev_data=dev_data,
                    batch_size=train_config["dev_batch_size"],
                )
                print(f"dev: iter {train_iter}, dev loss {dev_loss:.2f}")

                wandb.log(
                    {"dev_loss": dev_loss, "iteration": train_iter},
                    step=train_iter,
                )

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    patience = 0
                    print("Save currently the best model to [%s]" % model_save_path)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), f"{str(model_save_path)}.optim")

                elif patience < int(train_config["patience"]):
                    patience += 1
                    print(f"Hit patience {patience}")

                    if patience == int(train_config["patience"]):
                        num_trial += 1
                        print(f"Hit #{num_trial} trial")
                        if num_trial == int(train_config["max_num_trial"]):
                            print("Early stop!")
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]["lr"] * float(
                            train_config["lr_decay"]
                        )
                        print(
                            f"Load previously best model and decay learning rate to {lr:.6f}"
                        )

                        # load model
                        params = torch.load(
                            model_save_path,
                            map_location=lambda storage, loc: storage,
                        )
                        model.load_state_dict(params["state_dict"])
                        model = model.to(device)

                        print("Restore parameters of the optimizers")
                        optimizer.load_state_dict(
                            torch.load(f"{str(model_save_path)}.optim")
                        )

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        # reset patience
                        patience = 0

        if epoch == train_config["max_epoch"]:
            print("reached maximum number of epochs!")
            return


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to the model YAML config.",
    )

    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to the training YAML config.",
    )

    parser.add_argument(
        "--train-source-file",
        type=Path,
        default=Path("..", "multi30k_data", "train.en"),
        help="Path to the train source file, containing source sentences.",
    )

    parser.add_argument(
        "--train-target-file",
        type=Path,
        default=Path("..", "multi30k_data", "train.fr"),
        help="Path to the train target file, containing gold-standard target sentences.",
    )

    parser.add_argument(
        "--dev-source-file",
        type=Path,
        default=Path("..", "multi30k_data", "val.en"),
        help="Path to the dev source file, containing source sentences.",
    )

    parser.add_argument(
        "--dev-target-file",
        type=Path,
        default=Path("..", "multi30k_data", "val.fr"),
        help="Path to the dev target file, containing gold-standard target sentences.",
    )

    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=Path("vocab", "vocab.json"),
        help="Path to the vocabulary file.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("./models"),
        help="Path to the directory where checkpoints will be saved.",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="f21nl-cw2",
        help="Weights and Biases project name.",
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=f"nmt_{int(time.time())}",
        help="Weights and Biases run name.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    train(args)
