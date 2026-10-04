"""SageMaker training job: fine-tune BERT with a classification head, validation-only selection.

Standalone (no project imports): it runs inside the Hugging Face PyTorch training container.
Repeat 0 tries every learning rate in the grid for up to `max_epochs` epochs and keeps the
(learning rate, epoch) with the lowest validation log-loss; later repeats reuse that choice and
train for that many epochs. Each repeat writes class probabilities for its validation and test
examples, then refits the chosen configuration on the nested training fractions given in
`fractions.json` for the learning curve. Test labels are never read: they are not in the input.

Input  (/opt/ml/input/data/data):  job.json, repeat_<r>/{train,val,test}.jsonl, repeat_<r>/fractions.json
Output (/opt/ml/output/data):      predictions.csv, learning_curve.csv, run.json
"""
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

START = time.time()


def log(msg: str) -> None:
    print(f"bert: {msg} ({time.time() - START:.0f}s elapsed)", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Encoded:
    def __init__(self, tok, texts: list[str], max_length: int):
        full = tok(texts, truncation=False, add_special_tokens=True)["input_ids"]
        self.truncated = int(sum(len(x) > max_length for x in full))
        self.ids = tok(texts, truncation=True, max_length=max_length)["input_ids"]
        self.pad = tok.pad_token_id

    def __len__(self) -> int:
        return len(self.ids)

    def batch(self, idx, device):
        seqs = [self.ids[i] for i in idx]
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        return ids.to(device), mask.to(device)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, enc: Encoded, bs: int, device) -> tuple[np.ndarray, float]:
    model.eval()
    order = np.argsort([len(x) for x in enc.ids], kind="stable")      # similar lengths together: less padding
    out = np.zeros((len(enc), model.config.num_labels))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for s in range(0, len(order), bs):
        idx = order[s: s + bs]
        ids, mask = enc.batch(idx, device)
        with autocast(device):
            logits = model(input_ids=ids, attention_mask=mask).logits
        out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t


def logloss(y: np.ndarray, P: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def train(job, src, k, enc_tr, ytr, lr, epochs, seed, device, enc_val=None, yval=None):
    """Fine-tune for `epochs` epochs. With validation data, keep the epoch with the lowest validation log-loss."""
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(src, revision=job.get("revision"), num_labels=k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=job["weight_decay"])
    bs = job["batch_size"]
    steps = epochs * math.ceil(len(enc_tr) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(job["warmup_ratio"] * steps), steps)
    rng = np.random.default_rng(seed)
    history, best, t = [], None, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(enc_tr))
        for s in range(0, len(order), bs):
            idx = order[s: s + bs]
            ids, mask = enc_tr.batch(idx, device)
            with autocast(device):
                loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=mask).logits.float(),
                                                         torch.as_tensor(ytr[idx], device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
        if enc_val is not None:
            P, _ = predict(model, enc_val, job["eval_batch_size"], device)
            ll = logloss(yval, P)
            history.append({"epoch": epoch, "val_logloss": ll})
            if best is None or ll < best[0]:
                best = (ll, epoch, copy.deepcopy({k_: v.detach().cpu() for k_, v in model.state_dict().items()}))
    if best is not None:
        model.load_state_dict(best[2])
    return model, history, (best[1] if best else epochs), time.time() - t


def main(src: Path, out: Path) -> None:
    job = json.loads((src / "job.json").read_text())
    k, seed = job["n_classes"], job["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_src = os.environ.get("BERT_MODEL_DIR", job["model_id"])
    tok = AutoTokenizer.from_pretrained(model_src, revision=job.get("revision"))
    pcols = [f"p{c}" for c in range(k)]
    preds, curve, repeats = [], [], {}
    selected = None
    for r in range(job["n_repeats"]):
        d = src / f"repeat_{r}"
        tr, va, te = (read_jsonl(d / f"{s}.jsonl") for s in ("train", "val", "test"))
        if any("y" in x for x in te):
            raise RuntimeError("test examples must not carry labels")
        ytr, yval = np.array([x["y"] for x in tr]), np.array([x["y"] for x in va])
        enc = {s: Encoded(tok, [x["text"] for x in rows], job["max_length"]) for s, rows in (("train", tr), ("val", va), ("test", te))}
        info = {"truncated": {s: e.truncated for s, e in enc.items()}, "n": {s: len(e) for s, e in enc.items()}}
        if selected is None:                       # repeat 0: the grid, on validation log-loss only
            grid = []
            best = None
            for lr in job["grid"]["learning_rate"]:
                model, hist, epoch, secs = train(job, model_src, k, enc["train"], ytr, lr, job["max_epochs"], seed + r,
                                                 device, enc["val"], yval)
                ll = min(h["val_logloss"] for h in hist)
                grid.append({"learning_rate": lr, "history": hist, "best_epoch": epoch, "val_logloss": ll, "train_s": secs})
                log(f"repeat {r} lr {lr}: best epoch {epoch}, validation log-loss {ll:.4f}")
                if best is None or ll < best[0]:
                    best = (ll, lr, epoch, model, secs)
                else:
                    del model
            selected = {"learning_rate": best[1], "epochs": best[2], "val_logloss": best[0], "selected_on": "repeat 0 validation log-loss"}
            model, train_s = best[3], best[4]
            info.update(grid=grid, label_budget={"training": len(tr), "early_stopping_and_selection": len(va),
                                                 "calibration": 0, "threshold_selection": len(va) if k == 2 else 0})
        else:                                      # later repeats reuse the configuration chosen on repeat 0
            model, _, _, train_s = train(job, model_src, k, enc["train"], ytr, selected["learning_rate"], selected["epochs"],
                                         seed + r, device)
            info["label_budget"] = {"training": len(tr), "early_stopping_and_selection": 0, "calibration": 0,
                                    "threshold_selection": len(va) if k == 2 else 0}
        info["train_s"] = train_s
        for split, rows in (("val", va), ("test", te)):
            P, secs = predict(model, enc[split], job["eval_batch_size"], device)
            info[f"infer_{split}_s"] = secs
            preds += [{"id": x["id"], "repeat": r, "split": split, **dict(zip(pcols, p))} for x, p in zip(rows, P)]
            if split == "test":
                test_P = P
        info["infer_s_per_item"] = info["infer_test_s"] / max(len(te), 1)
        del model
        fracs = json.loads((d / "fractions.json").read_text())
        info["curve"] = {}
        for f in job["fractions"]:
            idx = np.array(fracs[str(f)], int)
            if f == 1.0 or len(idx) == len(tr):
                P = test_P
            elif time.time() - START > job["time_budget_s"]:
                info["curve"][str(f)] = {"skipped": "time budget used"}
                log(f"repeat {r} fraction {f}: skipped, time budget used")
                continue
            else:
                sub = Encoded.__new__(Encoded)
                sub.ids, sub.pad, sub.truncated = [enc["train"].ids[i] for i in idx], enc["train"].pad, 0
                m, _, _, secs = train(job, model_src, k, sub, ytr[idx], selected["learning_rate"], selected["epochs"],
                                      seed + r, device)
                P, _ = predict(m, enc["test"], job["eval_batch_size"], device)
                del m
            info["curve"][str(f)] = {"n_train": int(len(idx))}
            curve += [{"id": x["id"], "repeat": r, "fraction": f, "n_train": int(len(idx)), **dict(zip(pcols, p))}
                      for x, p in zip(te, P)]
            log(f"repeat {r} fraction {f} ({len(idx)} labels) done")
        repeats[str(r)] = info
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("predictions.csv", preds), ("learning_curve.csv", curve)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(), w.writerows(rows)
    import transformers
    run = {"dataset": job["dataset"], "selected": selected, "repeats": repeats, "fractions": job["fractions"],
           "max_length": job["max_length"], "model_id": job["model_id"], "revision": job.get("revision"),
           "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0]},
           "elapsed_s": time.time() - START}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    log("done")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/ml/input/data/data"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "/opt/ml/output/data"))
