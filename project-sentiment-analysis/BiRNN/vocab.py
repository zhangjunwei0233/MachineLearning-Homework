from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

SPECIAL_PAD = "<pad>"
SPECIAL_UNK = "<unk>"
SPECIAL_BOS = "<bos>"
SPECIAL_EOS = "<eos>"
SPECIAL_NUM = "<num>"
DEFAULT_SPECIALS = [SPECIAL_PAD, SPECIAL_UNK, SPECIAL_BOS, SPECIAL_EOS, SPECIAL_NUM]


class Vocab:
    def __init__(self, stoi: dict, itos: List[str]):
        self.stoi = stoi
        self.itos = itos
        self.pad_id = stoi[SPECIAL_PAD]
        self.unk_id = stoi[SPECIAL_UNK]
        self.bos_id = stoi[SPECIAL_BOS]
        self.eos_id = stoi[SPECIAL_EOS]
        self.num_id = stoi[SPECIAL_NUM]

    def __len__(self) -> int:
        return len(self.itos)

    def encode_tokens(
        self,
        tokens: Sequence[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> List[int]:
        pieces: List[int] = []
        if add_bos:
            pieces.append(self.bos_id)
        for tok in tokens:
            pieces.append(self.stoi.get(tok, self.unk_id))
        if add_eos:
            pieces.append(self.eos_id)
        if max_length is not None:
            pieces = pieces[:max_length]
        return pieces

    def decode_ids(self, ids: Sequence[int], skip_special: bool = True) -> List[str]:
        output: List[str] = []
        specials = {self.pad_id, self.unk_id, self.bos_id, self.eos_id, self.num_id} if skip_special else set()
        for idx in ids:
            if idx in specials:
                continue
            output.append(self.itos[idx] if 0 <= idx < len(self.itos) else SPECIAL_UNK)
        return output


def build_vocab(
    token_sequences: Iterable[Sequence[str]],
    min_freq: int = 2,
    max_size: int | None = None,
    specials: List[str] | None = None,
) -> Vocab:
    specials = specials or list(DEFAULT_SPECIALS)
    counter: Counter[str] = Counter()
    for seq in token_sequences:
        counter.update(seq)

    # Reserve space for specials
    most_common = counter.most_common()
    filtered = [tok for tok, freq in most_common if freq >= min_freq and tok not in specials]
    if max_size is not None:
        filtered = filtered[: max(0, max_size - len(specials))]

    itos = list(specials) + filtered
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)
