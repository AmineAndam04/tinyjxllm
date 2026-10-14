import json
import regex
from trainer import merge

GPT4_PATTERN = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|"""
    r"""[^\r\n\p{L}\p{N}]?+\p{L}++|"""
    r"""\p{N}{1,3}+|"""
    r"""[^\s\p{L}\p{N}]++[\r\n]*+|"""
    r"""\s*+\r?\n|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)


class Tokenizer:
    def __init__(self, files_path: str, special_tokens: None | list = None):
        # Load vocab, merges and special tokens
        self.load(files_path)
        vocab_size = max(self.vocab, default=-1) + 1
        # Update the special tokens only if it is unseen one
        if special_tokens:
            idx = 0
            for special_token in special_tokens:
                if special_token not in self.special_tokens:
                    self.vocab[vocab_size + idx] = special_token.encode("utf-8")
                    self.special_tokens[special_token] = vocab_size + idx
                    idx += 1

    def encode(self, text):
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if special_tokens:
            pattern = "(" + "|".join(regex.escape(token) for token in special_tokens) + ")"
            chunks = regex.split(pattern, text)
        else:
            chunks = [text]
        text = []

        for chunk in chunks:
            if chunk not in self.special_tokens:
                tokens = GPT4_PATTERN.findall(chunk)
                text.extend(tokens)
            else:
                text.extend([chunk])
        tokens = [
            (
                list(token.encode("utf-8"))
                if token not in special_tokens
                else [self.special_tokens[token]]
            )
            for token in text
        ]
        encoded = []
        for token in tokens:
            while len(token) >= 2:  # while you have something to merge
                pairs = list(zip(token, token[1:]))
                to_merge = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                if to_merge not in self.merges:
                    break
                token = merge(tokens=token, pair=to_merge, new_id=self.merges[to_merge])
            encoded.extend(token)
        return encoded

    def decode(self, tokens):
        text = [self.vocab[idx] for idx in tokens]
        text = b"".join(text)
        text = text.decode("utf-8", errors="replace")
        return text

    def load(self, files_path):
        with open(f"{files_path}/vocab.json", "r") as file:
            vocab = json.load(file)
            self.vocab = {int(idx): bytes.fromhex(token) for idx, token in vocab.items()}

        with open(f"{files_path}/merges.json", "r") as file:
            merges = json.load(file)
            self.merges = {(a, b): new_id for a, b, new_id in merges}

        with open(f"{files_path}/special_tokens.json", "r") as file:
            self.special_tokens = json.load(file)
