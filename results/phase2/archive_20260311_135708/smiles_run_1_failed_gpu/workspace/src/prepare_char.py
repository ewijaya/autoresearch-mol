"""
Shared utilities for character-level sequence datasets.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

TIME_BUDGET = 300
EVAL_TOKENS = 2_097_152

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = SPECIAL_TOKENS


def _device_or_default(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_file(cls, tokenizer_path: str | Path) -> "CharTokenizer":
        with open(tokenizer_path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(stoi=payload["stoi"], itos=payload["itos"])

    def save(self, tokenizer_path: str | Path) -> None:
        payload = {"stoi": self.stoi, "itos": self.itos}
        with open(tokenizer_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_vocab_size(self) -> int:
        return len(self.itos)

    def get_pad_token_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    def get_bos_token_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    def get_eos_token_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    def get_unk_token_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str) -> list[int]:
        unk_id = self.get_unk_token_id()
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        special = set(SPECIAL_TOKENS)
        chars = [self.itos[idx] for idx in ids if self.itos[idx] not in special]
        return "".join(chars)

    def token_bytes(self) -> torch.Tensor:
        values = []
        special = set(SPECIAL_TOKENS)
        for token in self.itos:
            values.append(0 if token in special else len(token.encode("utf-8")))
        return torch.tensor(values, dtype=torch.int32)


def build_tokenizer_from_texts(texts: Iterable[str]) -> CharTokenizer:
    charset = sorted({ch for text in texts for ch in text})
    itos = SPECIAL_TOKENS + charset
    stoi = {token: idx for idx, token in enumerate(itos)}
    return CharTokenizer(stoi=stoi, itos=itos)


def save_stream(
    output_path: str | Path,
    sequences: list[str],
    tokenizer: CharTokenizer,
) -> dict[str, int]:
    bos = tokenizer.get_bos_token_id()
    eos = tokenizer.get_eos_token_id()

    stream = []
    lengths = []
    for seq in sequences:
        encoded = tokenizer.encode(seq)
        lengths.append(len(encoded))
        stream.extend([bos, *encoded, eos])

    payload = {
        "stream": np.asarray(stream, dtype=np.int32),
        "num_sequences": len(sequences),
        "raw_token_count": int(sum(lengths)),
        "stream_token_count": int(len(stream)),
    }
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "num_sequences": payload["num_sequences"],
        "raw_token_count": payload["raw_token_count"],
        "stream_token_count": payload["stream_token_count"],
    }


def load_split(split_path: str | Path) -> dict:
    with open(split_path, "rb") as handle:
        return pickle.load(handle)


def make_stream_dataloader(
    split_path: str | Path,
    batch_size: int,
    seq_len: int,
    device: str | torch.device | None = None,
):
    device = _device_or_default(device)
    payload = load_split(split_path)
    stream_np = payload["stream"]
    if len(stream_np) <= seq_len + 1:
        raise ValueError(f"Split at {split_path} is too short for seq_len={seq_len}")

    stream = torch.from_numpy(stream_np.astype(np.int64, copy=False))
    pin_memory = device.type == "cuda"
    cpu_x = torch.empty((batch_size, seq_len), dtype=torch.long, pin_memory=pin_memory)
    cpu_y = torch.empty((batch_size, seq_len), dtype=torch.long, pin_memory=pin_memory)
    epoch = 1
    pos = 0

    while True:
        for row in range(batch_size):
            if pos + seq_len + 1 > stream.numel():
                pos = 0
                epoch += 1
            chunk = stream[pos:pos + seq_len + 1]
            cpu_x[row].copy_(chunk[:-1])
            cpu_y[row].copy_(chunk[1:])
            pos += seq_len

        if device.type == "cuda":
            yield cpu_x.to(device, non_blocking=True), cpu_y.to(device, non_blocking=True), epoch
        else:
            yield cpu_x.clone(), cpu_y.clone(), epoch


@torch.no_grad()
def evaluate_bpb(
    model,
    tokenizer: CharTokenizer,
    batch_size: int,
    seq_len: int,
    split_path: str | Path,
    eval_tokens: int = EVAL_TOKENS,
    device: str | torch.device | None = None,
) -> float:
    device = _device_or_default(device)
    token_bytes = tokenizer.token_bytes().to(device)
    loader = make_stream_dataloader(split_path, batch_size, seq_len, device=device)
    steps = max(1, eval_tokens // (batch_size * seq_len))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
