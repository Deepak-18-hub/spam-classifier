"""Fine-tune DistilBERT on the same split as the baseline and compare the two.

Run from the repo root:  python -m spamclf.train_bert
Needs requirements-m5.txt (PyTorch CUDA build + transformers) and a CUDA GPU.
This is a COMPARISON: the API and Docker image keep serving the baseline.
"""
import copy
import json
import platform
import random
import statistics
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from spamclf.data import SEED, load_split
from spamclf.model import MODEL_PATH, THRESHOLD, load_model, spam_scores

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = "distilbert-base-uncased"
WEIGHTS_DIR = ROOT / "local_models" / "distilbert_spam"  # git-ignored, ~255 MB
METRICS_PATH = ROOT / "models" / "metrics_bert.json"
BASELINE_METRICS = MODEL_PATH.with_name("metrics.json")

TRAIN_SEEDS = [42, 43, 44]  # seeds change training only; the data split is always SEED
EPOCHS, BATCH, LR, WEIGHT_DECAY, WARMUP, MAX_LEN, VAL_FRACTION = 3, 16, 2e-5, 0.01, 0.1, 128, 0.1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, texts):
    return tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


@torch.no_grad()
def predict_scores(model, tok, texts, device, batch=64):
    """Spam probability (softmax column 1) for each text."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = {k: v.to(device) for k, v in encode(tok, texts[i:i + batch]).items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def spam_prf(y_true, scores):
    pred = (scores >= THRESHOLD).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"precision": round(float(p[0]), 4), "recall": round(float(r[0]), 4), "f1": round(float(f[0]), 4),
            "average_precision": round(float(average_precision_score(y_true, scores)), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def train_one(seed, X_tr, y_tr, X_val, y_val, X_test, y_test, tok, device):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(device)
    # Class-weighted loss mirrors the baseline's class_weight="balanced" (n / (2 * class_count)).
    counts = np.bincount(y_tr, minlength=2)
    weights = torch.tensor(len(y_tr) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * ((len(X_tr) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(optim, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator().manual_seed(seed)
    y_tr_t = torch.tensor(y_tr)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_tr), generator=gen).tolist()
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = {k: v.to(device) for k, v in encode(tok, [X_tr[j] for j in idx]).items()}
            labels = y_tr_t[idx].to(device)
            optim.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        val_f1 = spam_prf(y_val, predict_scores(model, tok, X_val, device))["f1"]  # epoch chosen on VALIDATION only
        history.append({"epoch": epoch, "train_loss": round(total / len(X_tr), 4), "val_f1": val_f1})
        print(f"  seed {seed} epoch {epoch}: train loss {history[-1]['train_loss']} | val F1 {val_f1}")
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    train_seconds = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    model.load_state_dict(best_state)
    test_scores = predict_scores(model, tok, X_test, device)  # the test set is scored exactly once per run
    return model, test_scores, {"seed": seed, "history": history, "best_epoch": int(np.argmax([h["val_f1"] for h in history]) + 1),
                                "train_seconds": round(train_seconds, 1), "peak_vram_mb": round(peak_vram_mb),
                                "test": spam_prf(y_test, test_scores)}


def bootstrap_f1(y, pred_a, pred_b, n=2000, seed=0):
    """95% intervals for each model's spam F1 and for the paired difference (a minus b), resampling test messages."""
    rng = np.random.default_rng(seed)
    f1 = lambda yt, yp: precision_recall_fscore_support(yt, yp, labels=[1], zero_division=0)[2][0]
    fa, fb = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        fa.append(f1(y[i], pred_a[i]))
        fb.append(f1(y[i], pred_b[i]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"a_f1_95ci": ci(fa), "b_f1_95ci": ci(fb), "a_minus_b_95ci": ci(np.array(fa) - np.array(fb))}


def median_latency_ms(fn, texts, repeats=3):
    fn(texts[0])  # warm-up
    times = []
    for t in texts * repeats:
        t0 = time.perf_counter()
        fn(t)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available: check the PyTorch build and driver")
    device = torch.device("cuda")
    X_train, X_test, y_train, y_test, n_unique = load_split(SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=VAL_FRACTION, stratify=y_train, random_state=SEED)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)

    lens = [len(x) for x in tok(X_train + X_test)["input_ids"]]
    truncated = round(100 * float(np.mean(np.array(lens) > MAX_LEN)), 2)
    print(f"data: train {len(X_tr)} / val {len(X_val)} / test {len(X_test)} | tokens p50 {int(np.percentile(lens, 50))} "
          f"p99 {int(np.percentile(lens, 99))} max {max(lens)} | truncated at {MAX_LEN}: {truncated}%")

    # Prove the baseline is scored on the identical test set: its saved numbers must reproduce exactly.
    base_model = load_model()
    base_scores = spam_scores(base_model, X_test)
    base = spam_prf(y_test, base_scores)
    saved = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))["test"]["confusion"]
    if base["confusion"] != saved:
        raise RuntimeError(f"baseline does not reproduce its saved numbers: {base['confusion']} vs {saved}")
    print("baseline reproduced on this test set:", base["confusion"])

    runs, first = [], None
    for seed in TRAIN_SEEDS:
        model, scores, info = train_one(seed, X_tr, y_tr, X_val, y_val, X_test, y_test, tok, device)
        runs.append(info)
        print(f"seed {seed}: test {info['test']}")
        if first is None:
            first = (model, scores)
        else:
            del model
            torch.cuda.empty_cache()

    model, scores = first
    y = np.array(y_test)
    pred_b, pred_base = (scores >= THRESHOLD).astype(int), (base_scores >= THRESHOLD).astype(int)
    right_b, right_base = pred_b == y, pred_base == y
    paired = {"both_right": int((right_b & right_base).sum()), "bert_only_right": int((right_b & ~right_base).sum()),
              "baseline_only_right": int((~right_b & right_base).sum()), "both_wrong": int((~right_b & ~right_base).sum())}
    boot = bootstrap_f1(y, pred_b, pred_base)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(WEIGHTS_DIR)
    tok.save_pretrained(WEIGHTS_DIR)
    disk_mb = sum(f.stat().st_size for f in WEIGHTS_DIR.rglob("*") if f.is_file()) / 1024**2

    sample = X_test[:40]
    cpu_model = copy.deepcopy(model).to("cpu").float()
    bert_cpu_ms = median_latency_ms(lambda t: predict_scores(cpu_model, tok, [t], torch.device("cpu")), sample)
    base_ms = median_latency_ms(lambda t: spam_scores(base_model, [t]), sample)

    f1s = [r["test"]["f1"] for r in runs]
    result = {
        "trained_on": date.today().isoformat(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0)},
        "config": {"checkpoint": CHECKPOINT, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "warmup": WARMUP, "max_len": MAX_LEN, "loss": "class-weighted cross-entropy", "mixed_precision": "fp16 autocast",
                   "threshold": THRESHOLD, "epoch_chosen_on": "validation split of train", "train_seeds": TRAIN_SEEDS},
        "data": {"train": len(X_tr), "val": len(X_val), "test": len(X_test), "spam_in_test": int(y.sum()),
                 "tokens_truncated_pct": truncated},
        "baseline_test": base,
        "bert_runs": runs,
        "bert_test_f1_mean": round(statistics.mean(f1s), 4),
        "bert_test_f1_std": round(statistics.pstdev(f1s), 4),
        "paired_seed_%d_vs_baseline" % TRAIN_SEEDS[0]: paired,
        "bootstrap_seed_%d_bert_vs_baseline" % TRAIN_SEEDS[0]: boot,
        "cost_cpu": {"bert_median_ms_per_message": bert_cpu_ms, "baseline_median_ms_per_message": base_ms,
                     "bert_disk_mb": round(disk_mb, 1), "baseline_disk_kb": round(MODEL_PATH.stat().st_size / 1024)},
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("bert_test_f1_mean", "bert_test_f1_std", "cost_cpu")}, indent=2))
    print("paired:", paired, "\nbootstrap:", boot, "\nsaved", METRICS_PATH)


if __name__ == "__main__":
    main()
