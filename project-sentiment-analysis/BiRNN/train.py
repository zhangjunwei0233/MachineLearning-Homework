from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.serialization import add_safe_globals

from .data import SentimentDataset, collate_batch, load_tsv, stratified_split
from .model import SentimentRNNModel
from .tokenizer import tokenize
from .vocab import DEFAULT_SPECIALS, Vocab, build_vocab


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"decay": self.decay, "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.decay = state.get("decay", self.decay)
        shadow = state.get("shadow", {})
        self.shadow = {k: v.clone() for k, v in shadow.items()}
        self.backup = {}


def build_dataloaders(
    train_path: Path,
    vocab: Vocab,
    batch_size: int,
    val_ratio: float,
    seed: int,
    max_length: Optional[int],
    word_dropout: float,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    examples = load_tsv(train_path, is_train=True)
    train_ex, val_ex = stratified_split(examples, val_ratio=val_ratio, seed=seed)
    train_ds = SentimentDataset(train_ex, vocab=vocab, max_length=max_length)
    val_ds = SentimentDataset(val_ex, vocab=vocab, max_length=max_length)
    specials_set = {vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id, vocab.num_id}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(
            batch,
            pad_id=vocab.pad_id,
            unk_id=vocab.unk_id,
            word_dropout=word_dropout,
            specials=specials_set,
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(
            batch,
            pad_id=vocab.pad_id,
            unk_id=vocab.unk_id,
            word_dropout=0.0,
            specials=specials_set,
        ),
    )
    return train_loader, val_loader


def compute_class_weights(labels: List[int], num_classes: int = 5) -> torch.Tensor:
    counts = {i: 0 for i in range(num_classes)}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    total = sum(counts.values())
    weights = [total / max(1, counts[i]) for i in range(num_classes)]
    tensor = torch.tensor(weights, dtype=torch.float)
    tensor = tensor / tensor.mean()
    return tensor


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
    use_focal: bool,
    focal_gamma: float,
) -> torch.Tensor:
    if use_focal:
        # Focal loss with optional class weighting as alpha
        probs = torch.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_p_t = torch.log(p_t.clamp_min(1e-12))
        ce = -log_p_t
        alpha = class_weights[targets] if class_weights is not None else 1.0
        loss = (alpha * ((1.0 - p_t) ** focal_gamma) * ce).mean()
        return loss
    if label_smoothing > 0:
        num_classes = logits.size(-1)
        with torch.no_grad():
            smoothed = torch.full_like(logits, fill_value=label_smoothing / (num_classes - 1))
            smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        log_probs = torch.log_softmax(logits, dim=-1)
        weights = class_weights.unsqueeze(0) if class_weights is not None else 1.0
        loss = -(smoothed * log_probs) * weights
        return loss.sum(dim=1).mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion(logits, targets)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: SentimentGRUModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    ema: Optional[EMA],
    class_weights: Optional[torch.Tensor],
    use_focal: bool,
    focal_gamma: float,
    label_smoothing: float,
    clip_norm: float,
    device: torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for step, batch in enumerate(loader, start=1):
        input_ids, lengths, labels, _ = batch
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = compute_loss(logits, labels, class_weights, label_smoothing, use_focal, focal_gamma)
        loss.backward()
        if clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += batch_size

        if log_interval > 0 and step % log_interval == 0:
            running_acc = total_correct / total_samples if total_samples > 0 else 0.0
            running_loss = total_loss / total_samples if total_samples > 0 else 0.0
            print(
                f"  step {step:5d}/{len(loader):5d} | loss {running_loss:.4f} | acc {running_acc:.4f}"
            )
    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: SentimentGRUModel,
    loader: DataLoader,
    class_weights: Optional[torch.Tensor],
    use_focal: bool,
    focal_gamma: float,
    label_smoothing: float,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            input_ids, lengths, labels, _ = batch
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(input_ids, lengths)
            loss = compute_loss(logits, labels, class_weights, label_smoothing, use_focal, focal_gamma)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def predict(
    model: SentimentGRUModel,
    loader: DataLoader,
    device: torch.device,
) -> List[Tuple[int, int]]:
    model.eval()
    outputs: List[Tuple[int, int]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, lengths, _, phrase_ids = batch
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            logits = model(input_ids, lengths)
            preds = logits.argmax(dim=-1).cpu().tolist()
            outputs.extend(list(zip(phrase_ids.tolist(), preds)))
    return outputs


def save_submission(preds: List[Tuple[int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("PhraseId,Sentiment\n")
        for phrase_id, pred in preds:
            f.write(f"{phrase_id},{pred}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRU sentiment classifier (no LSTM/CNN/Transformer)")
    parser.add_argument("--train_path", type=Path, default=Path("data/train.tsv"))
    parser.add_argument("--test_path", type=Path, default=Path("data/test.tsv"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--embed_dim", type=int, default=360)
    parser.add_argument("--hidden_dim", type=int, default=448)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.025)
    parser.add_argument("--label_smoothing", type=float, default=0.025)
    parser.add_argument("--use_focal", action="store_true", help="Use focal loss instead of cross entropy")
    parser.add_argument("--focal_gamma", type=float, default=1.5, help="Gamma for focal loss (if enabled)")
    parser.add_argument("--word_dropout", type=float, default=0.025)
    parser.add_argument("--embed_dropout", type=float, default=0.15)
    parser.add_argument("--encoder_dropout", type=float, default=0.28)
    parser.add_argument("--warmup_ratio", type=float, default=0.12)
    parser.add_argument("--clip_norm", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between training logs")
    parser.add_argument("--resume_path", type=Path, default=None, help="Path to checkpoint to resume/skip train")
    parser.add_argument(
        "--skip_train_if_found",
        action="store_true",
        help="If a checkpoint exists, skip training and go straight to eval/test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = load_tsv(args.train_path, is_train=True)
    tokenized = [tokenize(ex.phrase) for ex in train_examples]
    default_vocab = build_vocab(tokenized, min_freq=2, specials=DEFAULT_SPECIALS)

    # Optionally load checkpoint before building model so vocab size matches
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / args.checkpoint
    ckpt_path = args.resume_path if args.resume_path is not None else best_path

    ckpt = None
    if ckpt_path.exists():
        add_safe_globals([Vocab])
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            print(f"Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt_path}: {e}")
            ckpt = None

    loaded_vocab = ckpt.get("vocab") if ckpt else None
    vocab = loaded_vocab or default_vocab

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        args.train_path,
        vocab,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_length=args.max_length,
        word_dropout=args.word_dropout,
        num_workers=args.num_workers,
    )

    model = SentimentRNNModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=5,
        embed_dropout=args.embed_dropout,
        encoder_dropout=args.encoder_dropout,
    ).to(device)

    loaded_model_state = ckpt.get("model_state") if ckpt else None
    loaded_epoch = ckpt.get("epoch", 0) if ckpt else 0
    loaded_best_val = ckpt.get("best_val_acc", 0.0) if ckpt else 0.0
    loaded_opt_state = ckpt.get("optimizer_state") if ckpt else None
    loaded_sched_state = ckpt.get("scheduler_state") if ckpt else None
    loaded_ema_state = ckpt.get("ema_state") if ckpt else None

    if loaded_model_state is not None:
        model.load_state_dict(loaded_model_state)

    labels_all = [ex.sentiment for ex in train_examples if ex.sentiment is not None]
    class_weights = compute_class_weights(labels_all).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    ema = EMA(model, decay=0.999) if args.use_ema else None

    if loaded_opt_state is not None:
        try:
            optimizer.load_state_dict(loaded_opt_state)
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")
    if loaded_sched_state is not None:
        try:
            scheduler.load_state_dict(loaded_sched_state)
        except Exception as e:
            print(f"Warning: could not load scheduler state: {e}")
    if ema is not None and loaded_ema_state is not None:
        try:
            ema.load_state_dict(loaded_ema_state)
        except Exception as e:
            print(f"Warning: could not load EMA state: {e}")

    best_val_acc = loaded_best_val
    epochs_no_improve = 0
    start_epoch = loaded_epoch + 1 if ckpt else 1

    if not (args.skip_train_if_found and ckpt is not None):
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                ema,
                class_weights,
                args.use_focal,
                args.focal_gamma,
                args.label_smoothing,
                args.clip_norm,
                device,
                args.log_interval,
            )
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                class_weights,
                args.use_focal,
                args.focal_gamma,
                args.label_smoothing,
                device,
            )
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab": vocab,
                        "epoch": epoch,
                        "best_val_acc": best_val_acc,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "ema_state": ema.state_dict() if ema is not None else None,
                    },
                    best_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print("Early stopping triggered")
                    break
    else:
        print(f"Checkpoint found at {ckpt_path}; skipping training as requested.")

    # Load best weights
    if best_path.exists():
        add_safe_globals([Vocab])
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])

    # Swap in EMA weights for eval if requested
    if ema is not None:
        ema.apply_shadow(model)

    if args.run_test:
        test_examples = load_tsv(args.test_path, is_train=False)
        test_ds = SentimentDataset(test_examples, vocab=vocab, max_length=args.max_length)
        specials_set = {vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id, vocab.num_id}
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_batch(
                batch,
                pad_id=vocab.pad_id,
                unk_id=vocab.unk_id,
                word_dropout=0.0,
                specials=specials_set,
            ),
        )
        preds = predict(model, test_loader, device)
        submission_path = args.output_dir / "submission.csv"
        save_submission(preds, submission_path)
        print(f"Saved submission to {submission_path}")

    if ema is not None:
        ema.restore(model)


if __name__ == "__main__":
    main()
