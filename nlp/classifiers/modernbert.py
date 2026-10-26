"""Fine-tuned ModernBERT classifier. torch/transformers are imported lazily."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ModernBertClassifier:
    name = "modernbert"

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_len: int = 512,
        epochs: int = 2,
        lr: float = 5e-5,
        batch_size: int = 16,
        seed: int = 20261026,
    ):
        self.model_name = model_name
        self.model_id = model_name
        self.max_len = max_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.classes_: list[str] = []
        self.model = None
        self.tok = None

    @staticmethod
    def _device():
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _tokenize(self, batch: list[str], dev):
        return self.tok(
            batch,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
        ).to(dev)

    def fingerprint(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_len": self.max_len,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def fit(self, texts: list[str], path_ids: list[str]) -> None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        torch.manual_seed(self.seed)
        self.classes_ = sorted(set(path_ids))
        if len(self.classes_) == 1:  # num_labels=1 would make HF train a regressor
            self.model = self.tok = None
            return
        col = {c: i for i, c in enumerate(self.classes_)}
        y = torch.tensor([col[p] for p in path_ids])
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.classes_),
            id2label=dict(enumerate(self.classes_)),
            label2id=col,
        )
        dev = self._device()
        use_amp = dev.type == "cuda"
        self.model.to(dev).train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        steps = self.epochs * -(-len(texts) // self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(len(texts))
            for s in range(0, len(order), self.batch_size):
                idx = order[s : s + self.batch_size]
                enc = self._tokenize([texts[i] for i in idx], dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    loss = self.model(**enc, labels=y[idx].to(dev)).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
        self.model.eval()

    def predict_proba(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        if self.model is None:
            return self.classes_, np.ones((len(texts), len(self.classes_)))
        import torch

        dev = self._device()
        self.model.to(dev).eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), self.batch_size):
                enc = self._tokenize(texts[s : s + self.batch_size], dev)
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        probs = np.vstack(out) if out else np.zeros((0, len(self.classes_)))
        return self.classes_, probs

    def save_pretrained(self, path: Path) -> None:
        path = Path(path)
        if self.model is not None:
            self.model.save_pretrained(path)
            self.tok.save_pretrained(path)
        (path / "meta.json").write_text(
            json.dumps({"classes": self.classes_, "model_name": self.model_name}),
            encoding="utf-8",
        )

    def load_pretrained(self, path: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.classes_ = meta["classes"]
        if len(self.classes_) == 1:
            self.model = self.tok = None
            return
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
