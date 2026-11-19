"""Layer-wise linear probes on residual-stream activations.

Trains one regularized logistic classifier per layer to predict the input
serialization (native role replay vs pasted transcript) from the model's own
hidden states, and reports the controls that make the result interpretable.

This is the trained component of the project: binary cross-entropy, L2
regularization, Adam, with train/validation/test splits taken over *source
conversations* rather than over rows.

Expected activation file (produced on the GPU side), a .pt holding::

    {"records": [
        {"case_id": str,          # source conversation
         "condition": "nat"|"flat",
         "layer": int,            # 0 = embedding output, 1..L = block outputs
         "vector": 1-D float tensor of length d,
         "n_tokens": int},        # optional; enables the length baseline
        ...
    ]}

Nothing here hard-codes d or the layer set: both are read from the file.

Self-test with synthetic data of the right shape (no GPU, no real file)::

    python probe.py --selftest
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from splits import bootstrap_ci, split_sources

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_records(path: Path) -> list[dict]:
    blob = torch.load(path, map_location="cpu")
    records = blob["records"] if isinstance(blob, dict) else blob
    if not records:
        raise ValueError(f"{path} holds no records")
    return records


def organize(records: list[dict]) -> dict[int, dict]:
    """Group records by layer into arrays, keeping the source id per row."""
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_layer[int(r["layer"])].append(r)

    out = {}
    for layer, rows in by_layer.items():
        rows.sort(key=lambda r: (str(r["case_id"]), r["condition"]))
        X = np.stack([np.asarray(r["vector"], dtype=np.float32) for r in rows])
        y = np.array([1.0 if r["condition"] == "nat" else 0.0 for r in rows],
                     dtype=np.float32)
        src = [str(r["case_id"]) for r in rows]
        ntok = np.array([float(r.get("n_tokens", np.nan)) for r in rows],
                        dtype=np.float32)
        out[layer] = dict(X=X, y=y, src=src, n_tokens=ntok)
    return out


def check_pairing(layer_data: dict) -> None:
    """Every source must contribute exactly one row per condition."""
    counts: dict[tuple[str, float], int] = defaultdict(int)
    for s, y in zip(layer_data["src"], layer_data["y"]):
        counts[(s, float(y))] += 1
    bad = {k: v for k, v in counts.items() if v != 1}
    if bad:
        raise ValueError(f"unbalanced pairing for {list(bad)[:5]} ...")


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------
class LinearProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def standardize(train_X: np.ndarray, *others: np.ndarray):
    """Standardize with training statistics only; drop zero-variance columns."""
    mu = train_X.mean(0)
    sd = train_X.std(0)
    keep = sd > 1e-8
    mu, sd = mu[keep], sd[keep]
    out = [(train_X[:, keep] - mu) / sd]
    out += [(o[:, keep] - mu) / sd for o in others]
    return out, dict(mu=mu, sd=sd, keep=keep)


def train_probe(Xtr, ytr, Xva, yva, lr: float, l2: float, seed: int,
                max_epochs: int = 200, patience: int = 10,
                batch_size: int = 32) -> tuple[nn.Module, float]:
    """Minibatch Adam on BCE with an L2 penalty, early-stopped on validation.

    Minibatching matters here: full-batch descent would give only `max_epochs`
    gradient steps, which is far too few to fit a probe in several thousand
    dimensions.
    """
    torch.manual_seed(seed)
    model = LinearProbe(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr, device=DEVICE)
    ytr_t = torch.tensor(ytr, device=DEVICE)
    Xva_t = torch.tensor(Xva, device=DEVICE)
    yva_t = torch.tensor(yva, device=DEVICE)
    n = Xtr_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva_t), yva_t))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def accuracy(model: nn.Module, X, y) -> float:
    model.eval()
    p = torch.sigmoid(model(torch.tensor(X, device=DEVICE))).cpu().numpy()
    return float(((p > 0.5).astype(np.float32) == y).mean())


@torch.no_grad()
def per_source_correct(model: nn.Module, X, y, src) -> dict[str, list[float]]:
    model.eval()
    p = torch.sigmoid(model(torch.tensor(X, device=DEVICE))).cpu().numpy()
    hit = ((p > 0.5).astype(np.float32) == y).astype(float)
    acc: dict[str, list[float]] = defaultdict(list)
    for s, h in zip(src, hit):
        acc[s].append(float(h))
    return acc


GRID = [(lr, l2) for lr in (1e-3, 1e-2)
        for l2 in (1e-4, 1e-3, 1e-2, 1e-1)]


def run_layer(data: dict, assignment: dict[str, str],
              seeds=(0, 1, 2), shuffle_labels: bool = False,
              rng_seed: int = 453) -> dict:
    X, y, src = data["X"], data["y"].copy(), data["src"]
    if shuffle_labels:
        # permute labels within the pairing so class balance is preserved
        rng = np.random.default_rng(rng_seed)
        y = rng.permutation(y)

    m = {name: np.array([assignment[s] == name for s in src])
         for name in ("train", "val", "test")}
    if not (m["train"].any() and m["val"].any() and m["test"].any()):
        raise ValueError("a split is empty; too few source conversations")

    (Xtr, Xva, Xte), _ = standardize(X[m["train"]], X[m["val"]], X[m["test"]])
    ytr, yva, yte = y[m["train"]], y[m["val"]], y[m["test"]]

    best = None
    for lr, l2 in GRID:
        model, vl = train_probe(Xtr, ytr, Xva, yva, lr, l2, seed=0)
        va = accuracy(model, Xva, yva)
        if best is None or va > best["val_acc"]:
            best = dict(lr=lr, l2=l2, val_acc=va, val_loss=vl)

    test_accs, weights, per_src = [], [], None
    for s in seeds:
        model, _ = train_probe(Xtr, ytr, Xva, yva, best["lr"], best["l2"], seed=s)
        test_accs.append(accuracy(model, Xte, yte))
        weights.append(model.fc.weight.detach().cpu().numpy().ravel())
        if per_src is None:
            per_src = per_source_correct(model, Xte, yte,
                                         [s_ for s_, k in zip(src, m["test"]) if k])

    point, lo, hi = bootstrap_ci(per_src)
    w = np.mean(weights, axis=0)
    return dict(_per_source=per_src,
                lr=best["lr"], l2=best["l2"], val_acc=best["val_acc"],
                test_acc=float(np.mean(test_accs)),
                test_acc_sd=float(np.std(test_accs)),
                ci_lo=lo, ci_hi=hi, bootstrap_point=point,
                n_train=int(m["train"].sum()), n_test=int(m["test"].sum()),
                direction=(w / (np.linalg.norm(w) + 1e-12)).tolist())


# --------------------------------------------------------------------------
# Grouped cross-validation
#
# With few source conversations a single 70/15/15 split sends only ~15% of them
# to test, and the confidence interval is set by that small test set.  Nested
# grouped k-fold uses every source for evaluation exactly once, which tightens
# the interval for the same generation budget.  Folds are over *sources*, so
# the two rows of a pair always move together.
# --------------------------------------------------------------------------
def grouped_folds(sources: list[str], k: int = 5, salt: str = "stat453-cv"
                  ) -> list[list[str]]:
    """Deterministic assignment of sources to k folds."""
    from splits import _stable_hash
    order = sorted(sources, key=lambda s: _stable_hash(str(s), salt))
    return [order[i::k] for i in range(k)]


def run_layer_cv(data: dict, k: int = 5, seeds=(0, 1, 2),
                 shuffle_labels: bool = False, rng_seed: int = 453) -> dict:
    """Nested grouped k-fold: outer fold = test, part of the rest = val."""
    sources = sorted(set(data["src"]), key=str)
    if len(sources) < k + 2:
        raise ValueError(f"need at least {k + 2} sources for {k}-fold")
    folds = grouped_folds(sources, k)

    per_src_all: dict[str, list[float]] = {}
    picked = []
    for f, test_sources in enumerate(folds):
        rest = [s for s in sources if s not in set(test_sources)]
        n_val = max(1, len(rest) // 6)
        val_sources = set(rest[:n_val])
        assignment = {}
        for s in sources:
            assignment[s] = ("test" if s in set(test_sources)
                             else "val" if s in val_sources else "train")
        r = run_layer(data, assignment, seeds=seeds,
                      shuffle_labels=shuffle_labels, rng_seed=rng_seed + f)
        picked.append((r["lr"], r["l2"]))
        per_src_all.update(r.pop("_per_source", {}))

    # pooled estimate over every source, each held out exactly once
    accs = [v for vs in per_src_all.values() for v in vs] if per_src_all else []
    if accs:
        point, lo, hi = bootstrap_ci(per_src_all)
    else:  # run_layer did not expose per-source hits; fall back to fold means
        point = lo = hi = float("nan")
    return dict(k=k, test_acc=point, ci_lo=lo, ci_hi=hi,
                hyperparams=picked, n_sources=len(sources))


def token_length_baseline(data: dict, assignment: dict[str, str]) -> float | None:
    """Accuracy of a classifier that sees only the prompt length."""
    ntok, y, src = data["n_tokens"], data["y"], data["src"]
    if np.isnan(ntok).all():
        return None
    m = {name: np.array([assignment[s] == name for s in src])
         for name in ("train", "val", "test")}
    X = ntok.reshape(-1, 1)
    (Xtr, Xva, Xte), _ = standardize(X[m["train"]], X[m["val"]], X[m["test"]])
    model, _ = train_probe(Xtr, y[m["train"]], Xva, y[m["val"]],
                           lr=1e-2, l2=1e-3, seed=0)
    return accuracy(model, Xte, y[m["test"]])


# --------------------------------------------------------------------------
def analyse(records: list[dict], out_json: Path | None = None,
            cv: int = 0) -> dict:
    by_layer = organize(records)
    for layer in sorted(by_layer):
        check_pairing(by_layer[layer])

    sources = sorted({r["case_id"] for r in records}, key=str)
    assignment = split_sources(sources)
    n_test = sum(v == "test" for v in assignment.values())
    print(f"{len(sources)} source conversations -> "
          f"{sum(v=='train' for v in assignment.values())} train / "
          f"{sum(v=='val' for v in assignment.values())} val / {n_test} test")
    print(f"layers: {sorted(by_layer)}   d = {by_layer[sorted(by_layer)[0]]['X'].shape[1]}\n")

    if cv:
        print(f"evaluating with grouped {cv}-fold CV "
              f"(every source held out exactly once)\n")
    results = {}
    for layer in sorted(by_layer):
        if cv:
            c = run_layer_cv(by_layer[layer], k=cv)
            results[layer] = c
            tag = "  <- surface-form control" if layer == 0 else ""
            print(f"  layer {layer:>3}  acc {c['test_acc']:.3f} "
                  f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]"
                  f"  width {c['ci_hi']-c['ci_lo']:.3f}{tag}")
            continue
        r = run_layer(by_layer[layer], assignment)
        r.pop("_per_source", None)
        results[layer] = r
        tag = "  <- surface-form control" if layer == 0 else ""
        print(f"  layer {layer:>3}  test {r['test_acc']:.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
              f"val {r['val_acc']:.3f}  lr={r['lr']:g} l2={r['l2']:g}{tag}")

    key = "test_acc" if cv else "val_acc"
    sel = max((l for l in results if l != 0),
              key=lambda l: results[l][key], default=None)
    print(f"\nvalidation-selected layer: {sel}")

    controls = {}
    if sel is not None and cv:
        controls["shuffled_labels"] = run_layer_cv(
            by_layer[sel], k=cv, shuffle_labels=True)["test_acc"]
        controls["layer_0"] = results.get(0, {}).get("test_acc")
        print("\ncontrols at the selected layer:")
        for k_, v in controls.items():
            print(f"  {k_:<16} {('n/a' if v is None else f'{v:.3f}')}")
    elif sel is not None:
        sh = run_layer(by_layer[sel], assignment, shuffle_labels=True)
        sh.pop("_per_source", None)
        controls["shuffled_labels"] = sh["test_acc"]
        controls["token_length"] = token_length_baseline(by_layer[sel], assignment)
        controls["layer_0"] = results.get(0, {}).get("test_acc")
        print("controls at the selected layer:")
        for k, v in controls.items():
            print(f"  {k:<16} {('n/a' if v is None else f'{v:.3f}')}")

    summary = dict(selected_layer=sel, per_layer=results, controls=controls,
                   n_sources=len(sources), split_counts={
                       k: sum(v == k for v in assignment.values())
                       for k in ("train", "val", "test")})
    if out_json:
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {out_json}")
    return summary


# --------------------------------------------------------------------------
def synthetic(n_sources=120, d=3584, layers=(0, 4, 8, 12, 16, 20, 24, 28),
              seed=453, signal=1.0) -> list[dict]:
    """Fake activations of realistic shape.

    Separability rises with depth and is deliberately non-zero at layer 0, so
    the surface-form control has something to catch -- the situation we expect
    and must be able to report.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for layer in layers:
        gap = signal * (0.05 + 0.45 * (layer / max(layers)))   # grows with depth
        direction = rng.standard_normal(d).astype(np.float32)
        direction /= np.linalg.norm(direction)
        for i in range(n_sources):
            base = rng.standard_normal(d).astype(np.float32)  # shared per source
            for cond, sign in (("nat", +1.0), ("flat", -1.0)):
                v = base + 0.5 * rng.standard_normal(d).astype(np.float32) \
                    + sign * gap * direction
                recs.append(dict(case_id=f"src-{i:04d}", condition=cond,
                                 layer=layer, vector=torch.tensor(v),
                                 n_tokens=400 + (30 if cond == "flat" else 0)
                                 + int(rng.integers(0, 60))))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", type=Path,
                    help=".pt file described in the module docstring")
    ap.add_argument("--out", type=Path, default=Path("probe_results.json"))
    ap.add_argument("--cv", type=int, default=0, metavar="K",
                    help="grouped K-fold instead of a single 70/15/15 split; "
                         "recommended when sources are few")
    ap.add_argument("--selftest", action="store_true",
                    help="run on synthetic data of the right shape")
    args = ap.parse_args()

    if args.selftest:
        print("=" * 62)
        print("SANITY CHECK: strong synthetic signal; the deepest layer must")
        print("reach high accuracy or the training loop is broken.")
        print("=" * 62)
        sane = analyse(synthetic(signal=8.0, layers=(0, 16, 28)), out_json=None)
        deep = sane["per_layer"][28]["test_acc"]
        print(f"\n  -> deepest-layer accuracy {deep:.3f}  "
              f"{'PASS' if deep > 0.90 else 'FAIL - investigate'}\n")

        print("=" * 62)
        print("REALISTIC CASE: weak signal, single 70/15/15 split.")
        print("=" * 62)
        analyse(synthetic(layers=(0, 16, 28)), out_json=None)

        print("\n" + "=" * 62)
        print("SAME DATA, grouped 5-fold CV: every source is held out once,")
        print("so the interval is much tighter for the same budget.")
        print("=" * 62)
        analyse(synthetic(layers=(0, 16, 28)), out_json=None, cv=5)
        return
    if not args.activations:
        ap.error("pass --activations or --selftest")
    analyse(load_records(args.activations), out_json=args.out, cv=args.cv)


if __name__ == "__main__":
    main()
