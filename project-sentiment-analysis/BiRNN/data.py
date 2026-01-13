from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import tokenize
from .vocab import Vocab


@dataclass
class PhraseExample:
    phrase_id: int
    sentence_id: int
    phrase: str
    sentiment: Optional[int] = None


class SentimentDataset(Dataset):
    def __init__(
        self,
        examples: List[PhraseExample],
        vocab: Vocab,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ):
        self.examples = examples
        self.vocab = vocab
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        tokens = tokenize(ex.phrase)
        ids = self.vocab.encode_tokens(tokens, add_bos=self.add_bos, add_eos=self.add_eos, max_length=self.max_length)
        return ids, ex.sentiment if ex.sentiment is not None else -1, ex.phrase_id


def _apply_word_dropout(ids: Sequence[int], pad_id: int, unk_id: int, drop_prob: float, specials: set) -> List[int]:
    if drop_prob <= 0:
        return list(ids)
    dropped = []
    for tok_id in ids:
        if tok_id in specials or tok_id == pad_id:
            dropped.append(tok_id)
        else:
            if random.random() < drop_prob:
                dropped.append(unk_id)
            else:
                dropped.append(tok_id)
    return dropped


def collate_batch(
    batch: List[Tuple[Sequence[int], int, int]],
    pad_id: int,
    unk_id: int,
    word_dropout: float = 0.0,
    specials: Optional[set] = None,
):
    specials = specials or set()
    sequences, labels, phrase_ids = zip(*batch)
    sequences = [_apply_word_dropout(seq, pad_id, unk_id, word_dropout, specials) for seq in sequences]
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    batch_size = len(sequences)

    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    labels_tensor = torch.tensor(labels, dtype=torch.long) if labels[0] != -1 else None
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    phrase_ids_tensor = torch.tensor(phrase_ids, dtype=torch.long)
    return padded, lengths_tensor, labels_tensor, phrase_ids_tensor


def load_tsv(path: Path, is_train: bool = True) -> List[PhraseExample]:
    examples: List[PhraseExample] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            phrase_id = int(row["PhraseId"])
            sentence_id = int(row["SentenceId"])
            phrase = row["Phrase"]
            sentiment = int(row["Sentiment"]) if is_train else None
            examples.append(PhraseExample(phrase_id, sentence_id, phrase, sentiment))
    return examples


def stratified_split(examples: List[PhraseExample], val_ratio: float, seed: int = 13) -> Tuple[List[PhraseExample], List[PhraseExample]]:
    rng = random.Random(seed)
    buckets = {}
    for ex in examples:
        buckets.setdefault(ex.sentiment, []).append(ex)
    train, val = [], []
    for label, items in buckets.items():
        rng.shuffle(items)
        cut = int(len(items) * (1 - val_ratio))
        train.extend(items[:cut])
        val.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
