import re
from typing import List

# Regex tokenizer: keeps contractions, numbers, emoticons, and punctuation tokens.
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[:;]-?[()D]|[^\s]")


def normalize_text(text: str) -> str:
    """Lowercase and collapse long punctuation runs."""
    lowered = text.lower()
    collapsed = re.sub(r"([!?.]){2,}", r"\1", lowered)
    return collapsed


def tokenize(text: str) -> List[str]:
    """Tokenize text while normalizing numbers to <num>."""
    text = normalize_text(text)
    tokens = TOKEN_PATTERN.findall(text)
    normalized = []
    for tok in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            normalized.append("<num>")
        else:
            normalized.append(tok)
    return normalized
