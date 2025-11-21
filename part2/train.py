import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import wandb
from model import GPT, GPTConfig
from utils import batch_iter, read_corpus
from vocab import Vocab


def get_lr_cosine_schedule_with_warmup(
    current_iteration: int,
    warmup_iters: int,
    lr_decay_iters: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate scheduler with warmup.

    Increases the learning rate linearly from 0 to max_lr over warmup_iters training iterations,
    then decreases it following a cosine curve down to min_lr over lr_decay_iters - warmup_iters
    iterations.

    Args:
        current_iteration (int): The current training iteration.
        warmup_iters (int): The number of iterations to linearly increase the learning rate.
        lr_decay_iters (int): The total number of iterations for learning rate decay.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.

    Returns:
        float: The learning rate for the current iteration.
    """
    # 1) linear warmup for warmup_iters steps
    if current_iteration < warmup_iters:
        return max_lr * (current_iteration + 1) / (warmup_iters + 1)

    # 2) if current_iteration > lr_decay_iters, return min learning rate
    if current_iteration > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (current_iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model: GPT, train_config: dict) -> torch.optim.Optimizer:
    """Configure the AdamW optimizer from the training configuration.

    Args:
        model (GPT): The GPT model to optimize.
        train_config (dict): The training configuration dictionary.

    Returns:
        torch.optim.Optimizer: The configured AdamW optimizer.
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": train_config["weight_decay"]},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    optimizer = torch.optim.AdamW(
        optim_groups, lr=train_config["lr"], betas=train_config["betas"]
    )

    return optimizer


def train(args: argparse.Namespace) -> None:
    """Train the NMT Model."""
    torch.manual_seed(0)
    np.random.seed(0)

    train_data = read_corpus(
        src_file_path=args.train_source_file,
        tgt_file_path=args.train_target_file,
        model_path=str(args.sentence_piece_model_file),
    )

    dev_data = read_corpus(
        src_file_path=args.dev_source_file,
        tgt_file_path=args.dev_target_file,
        model_path=str(args.sentence_piece_model_file),
    )

    vocab = Vocab.load(args.vocab_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_config = GPTConfig.from_yaml(args.model_config)
    model = GPT(model_config)

    if args.finetune_checkpoint_path:
        print(f"Load finetune checkpoint from {args.finetune_checkpoint_path}")
        state_dict = torch.load(args.finetune_checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    print(model)
    model = model.to(device)

    train_config = yaml.safe_load(args.train_config.read_text())

    optimizer = configure_optimizer(model=model, train_config=train_config)

    max_iters = train_config["max_epoch"] * (
        len(train_data) // train_config["train_batch_size"]
    )
    warmup_iters = int(train_config["warmup_percent_iters"] * max_iters)
    lr_decay_iters = max_iters

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = (
        report_tgt_words
    ) = 0
    cum_examples = report_examples = epoch = valid_num = 0
    train_time = begin_time = time.time()
    best_val_loss = float("inf")
    report_loss = 0.0
    patience = train_config["patience"]
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**model_config.to_dict(), **train_config},
    )
    model.train()
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)
    while True:
        epoch += 1
        for src_sents, tgt_sents in batch_iter(
            train_data, batch_size=train_config["train_batch_size"], shuffle=True
        ):
            optimizer.zero_grad()

            batch_size = len(src_sents)

            # Convert list of lists into tensors with padding
            inputs, targets = vocab.src.to_input_tensor(
                src_sents, tgt_sents, device=device
            )

            # outputs, loss = model(input_ids=inputs, targets=targets)
            model_output = model(input_ids=inputs, targets=targets)
            loss = model_output.loss

            loss.backward()

            # determine and set the learning rate for this iteration
            lr = (
                get_lr_cosine_schedule_with_warmup(
                    current_iteration=train_iter,
                    warmup_iters=warmup_iters,
                    lr_decay_iters=lr_decay_iters,
                    max_lr=train_config["lr"],
                    min_lr=train_config["min_lr"],
                )
                if train_config["decay_lr"]
                else train_config["lr"]
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

            report_loss += loss.item()

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % train_config["log_every"] == 0 and train_iter > 0:
                avg_loss = report_loss / train_config["log_every"]
                # print(
                #     f"iter {train_iter}: train loss {avg_loss:.4f}, current lr {lr:.6e}"
                # )
                print(
                    f"epoch {epoch}, iter {train_iter}, avg. loss {avg_loss:.2f} "
                    f"cum. examples {cum_examples}, speed {report_tgt_words / (time.time() - train_time):.2f} words/sec, time elapsed {time.time() - begin_time:.2f} sec"
                )
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.0 # type: ignore
                report_loss = 0.0
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                        "iteration": train_iter,
                    },
                    step=train_iter,
                )

            # perform validation
            if train_iter % train_config["valid_niter"] == 0 and train_iter > 0:
                model.eval()
                with torch.inference_mode():
                    val_loss = 0.0
                    num_val_batches = 0
                    for src_sents, tgt_sents in batch_iter(
                        dev_data, batch_size=train_config["dev_batch_size"]
                    ):
                        inputs, targets = vocab.src.to_input_tensor(
                            src_sents, tgt_sents, device=device
                        )

                        # outputs, loss = model(input_ids=inputs, targets=targets)
                        model_output = model(input_ids=inputs, targets=targets)
                        loss = model_output.loss
                        val_loss += loss.item()
                        num_val_batches += 1

                    val_loss /= num_val_batches
                    print(f"iter {train_iter}: validation loss {val_loss:.4f}")
                    wandb.log({"dev_loss": val_loss}, step=train_iter)

                    if val_loss < best_val_loss:
                        print(
                            f"new best validation loss {val_loss:.4f} (previous best {best_val_loss:.4f}) saving model {args.checkpoint_path}"
                        )
                        best_val_loss = val_loss
                        patience = train_config["patience"]
                        model.save(args.checkpoint_path)
                        # Also save the optimizer
                        torch.save(
                            optimizer.state_dict(),
                            str(Path(args.checkpoint_path, "gpt.model.optim")),
                        )
                    else:
                        patience -= 1
                        print(f"patience {patience}")
                        if patience == 0:
                            print("Ran out of patience, stopping training.")
                            return

                model.train()
            train_iter += 1

        if epoch == train_config["max_epoch"]:
            print("reached maximum number of epochs!")
            return

    print("Training complete!")
    print("Best validation loss: {:.4f}".format(best_val_loss))


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
        "--sentence-piece-model-file",
        type=Path,
        default=Path("vocab", "spm.model"),
        help="Path to the SentencePiece model file.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("models"),
        help="Path to the directory where checkpoints will be saved.",
    )

    parser.add_argument(
        "--finetune-checkpoint-path",
        type=Path,
        default=None,
        help="Path to a finetuning checkpoint.",
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
        default=f"gpt_{int(time.time())}",
        help="Weights and Biases run name.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
