"""Char-level tokenizer with byte fallback. Vocab = 657 chars seen in
train_corpus.txt (ids 0..656) + 256 raw-byte fallback ids (657..912), so
ANY UTF-8 text still round-trips exactly even if it uses codepoints never
seen in training.

You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
(e.g., BPE), as long as:
  1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
     LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
     graders both verify this round-trip — a lossy tokenizer makes bpb
     meaningless and disqualifies the run.
  2. this file keeps exposing:  load() -> tokenizer object with
     .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
     train.py and evaluate.py call load() with NO arguments — keep any
     extra parameters optional.
  3. anything it needs is saved under your submission folder and loaded by
     load() with no internet. Grading runs with cwd = your folder; resolve
     saved files relative to __file__ to be safe.
"""
import json
import os

_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "char_vocab.json")


class CharByteTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.char2id = {c: i for i, c in enumerate(chars)}
        self.byte_offset = len(chars)
        self.vocab_size = len(chars) + 256

    def encode(self, text):
        ids = []
        for c in text:
            i = self.char2id.get(c)
            if i is not None:
                ids.append(i)
            else:
                ids.extend(self.byte_offset + b for b in c.encode("utf-8"))
        return ids

    def decode(self, ids):
        out = []
        byte_run = bytearray()

        def flush():
            if byte_run:
                out.append(bytes(byte_run).decode("utf-8", errors="replace"))
                byte_run.clear()

        for i in ids:
            if i < self.byte_offset:
                flush()
                out.append(self.chars[i])
            else:
                byte_run.append(i - self.byte_offset)
        flush()
        return "".join(out)


def load(path=None):
    """Return the tokenizer used by evaluate.py. Replace as needed."""
    chars = json.load(open(path or _VOCAB_PATH, encoding="utf-8"))
    return CharByteTokenizer(chars)
