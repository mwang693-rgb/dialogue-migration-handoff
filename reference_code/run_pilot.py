"""Small, frozen-protocol dialogue-format steering pilot (not a formal test).

N/P share a fixed target-model-generated history. D/G/U are pure forward methods.
No future query enters extraction. Only the final prefill position and subsequent
cached decoding states are modified at one post-decoder-block residual location.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot_data import check_json, make_episodes, self_check


SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
CONTINUE_HINT = ("The transcript above is the ongoing conversation. Continue from its current "
                 "state, retaining valid earlier requirements and applying the latest user's updates.")


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def js(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = (p + q) / 2
    return float((np.sum(p * (np.log(p.clip(1e-300)) - np.log(m.clip(1e-300)))) +
                  np.sum(q * (np.log(q.clip(1e-300)) - np.log(m.clip(1e-300))))) / 2)


def render(history):
    return "\n\n".join(m["role"].capitalize() + ":\n" + m["content"] for m in history)


def messages(history, query=None, pasted=False, hint=False):
    if not pasted:
        return [SYSTEM] + history + ([{"role": "user", "content": query}] if query is not None else [])
    text = render(history)
    if hint:
        text += "\n\n" + CONTINUE_HINT
    if query is not None:
        text += "\n\nLatest user message:\n" + query
    return [SYSTEM, {"role": "user", "content": text}]


@contextmanager
def intervention(layer, vector=None, alpha=1.0):
    if vector is None:
        yield
        return
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if alpha == 0:
            return output
        adjusted = h.clone()
        adjusted[:, -1, :] += (vector * alpha).to(device=h.device, dtype=h.dtype)
        return (adjusted,) + output[1:] if isinstance(output, tuple) else adjusted
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def build_basis(deltas, rank):
    mean = deltas.mean(0)
    _, _, pcs = torch.linalg.svd(deltas - mean, full_matrices=False)
    candidates = [mean] + list(pcs)
    vectors = []
    for x in candidates:
        v = x.clone()
        # Reorthogonalization for numerical stability.
        for _ in range(2):
            for u in vectors:
                v -= torch.dot(u, v) * u
        if v.norm() > 1e-6:
            vectors.append(v / v.norm())
        if len(vectors) == rank:
            break
    if not vectors:
        raise ValueError("No nonzero basis direction")
    return mean, torch.stack(vectors, dim=1)


def kernel_checks():
    h = torch.zeros(1, 3, 4)
    layer = torch.nn.Identity()
    with intervention(layer, torch.ones(4), 1):
        actual = layer(h)
    assert torch.equal(actual[:, :-1], h[:, :-1]) and torch.equal(actual[:, -1], torch.ones(1, 4))
    with intervention(layer, torch.ones(4), 0):
        assert torch.equal(layer(h), h)
    assert len(layer._forward_hooks) == 0
    assert js([.1, .2, .7], [.1, .2, .7]) == 0
    assert abs(js([.1, .2, .7], [.7, .2, .1]) - js([.7, .2, .1], [.1, .2, .7])) < 1e-12
    a = torch.randn(8, 16)
    mean, basis = build_basis(a, 4)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-5)
    assert torch.allclose(basis @ (basis.T @ mean), mean, atol=1e-5)
    print("kernel invariants passed", flush=True)


class Runner:
    def __init__(self, snapshot, layer_index, max_input):
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        ).to("cuda").eval()
        self.model.requires_grad_(False)
        self.layer = self.model.model.layers[layer_index]
        self.layer_index = layer_index
        self.max_input = max_input
        self.load_seconds = time.perf_counter() - start
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def tokenize(self, msgs):
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_input:
            raise ValueError(f"Input exceeds paired budget: {len(ids)} > {self.max_input}")
        return text, ids

    @torch.inference_mode()
    def logits(self, msgs, vector=None, alpha=1.0):
        _, ids = self.tokenize(msgs)
        inputs = torch.tensor([ids], device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with intervention(self.layer, vector, alpha):
            output = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return output.logits[0, -1].float(), elapsed, len(ids)

    @torch.inference_mode()
    def extract(self, history):
        states, audit = [], []
        for pasted in [False, True]:
            msgs = messages(history, pasted=pasted)
            text, ids = self.tokenize(msgs)
            captured = []
            def capture(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                captured.append(h[0, -1].float().clone())
            handle = self.layer.register_forward_hook(capture)
            try:
                self.logits(msgs)
            finally:
                handle.remove()
            assert len(captured) == 1
            states.append(captured[0])
            audit.append(dict(format="P_H" if pasted else "N_H", rendered=text,
                              token_ids=ids, terminal_token_id=ids[-1], query_included=False))
        assert audit[0]["terminal_token_id"] == audit[1]["terminal_token_id"]
        return states[0] - states[1], audit

    @torch.inference_mode()
    def generate(self, msgs, max_tokens, vector=None, alpha=1.0):
        _, ids = self.tokenize(msgs)
        inp = torch.tensor([ids], device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with intervention(self.layer, vector, alpha):
            out = self.model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                                      do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                                      pad_token_id=self.tokenizer.eos_token_id, logits_to_keep=1)
        torch.cuda.synchronize()
        generated = out[0, len(ids):]
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        natural_end = len(generated) > 0 and int(generated[-1]) in eos_ids
        return dict(text=self.tokenizer.decode(generated, skip_special_tokens=True),
                    generated_tokens=len(generated), input_tokens=len(ids),
                    truncated=not natural_end and len(generated) >= max_tokens,
                    seconds=time.perf_counter() - start)

    def label_ids(self, msgs):
        text, prefix = self.tokenize(msgs)
        labels = []
        for label in "ABC":
            ids = self.tokenizer.encode(text + label, add_special_tokens=False)
            assert ids[:-1] == prefix and len(ids) == len(prefix) + 1, "Not a single token at actual boundary"
            labels.append(ids[-1])
        assert len(set(labels)) == 3
        return labels


def summarize(outdir):
    choices = read_jsonl(outdir / "choices.jsonl")
    free = read_jsonl(outdir / "free.jsonl")
    summary = {"status": "exploratory pilot, not locked test", "choice_rows": len(choices),
               "free_rows": len(free), "splits": {}}
    for split in ["dev", "validation"]:
        ss = {}
        for method in sorted({x["method"] for x in choices}):
            rs = [x for x in choices if x["split"] == split and x["method"] == method]
            fs = [x for x in free if x["split"] == split and x["method"] == method]
            if not rs:
                continue
            ss[method] = dict(n_histories=len({x["episode_id"] for x in rs}),
                              action_accuracy=float(np.mean([x["correct"] for x in rs])),
                              native_agreement=float(np.mean([x["native_agreement"] for x in rs])),
                              js_to_native=float(np.mean([x["js_native"] for x in rs])),
                              next_token_kl=float(np.mean([x["next_token_kl"] for x in rs])),
                              label_mass=float(np.mean([x["label_mass"] for x in rs])),
                              free_all_fields=float(np.mean([x["checker"]["all_fields"] for x in fs])) if fs else None,
                              free_truncated=sum(x["truncated"] for x in fs),
                              mean_forward_seconds=float(np.mean([x["forward_seconds"] for x in rs])))
        p_js = ss.get("P", {}).get("js_to_native", 0)
        for value in ss.values():
            value["js_restoration"] = 1 - value["js_to_native"] / p_js if p_js > 1e-6 else None
        summary["splits"][split] = ss
    write_json(outdir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/pilot_v1")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--dev-count", type=int, default=20)
    parser.add_argument("--layer", type=int, default=16, help="one-based decoder block")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-input", type=int, default=2048)
    parser.add_argument("--history-tokens", type=int, default=128)
    parser.add_argument("--free-tokens", type=int, default=128)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    self_check()
    kernel_checks()
    if args.self_check_only:
        return
    assert 1 < args.dev_count < args.count
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model_info = json.loads(Path("artifacts/model_snapshot.json").read_text())
    config = {**vars(args), **model_info, "dtype": "bfloat16", "decode": "greedy",
              "extraction": "history only, terminal assistant prefix", "seed": 453,
              "intervention": "last prompt token and cached generated tokens",
              "code_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), Path(__file__).with_name("pilot_data.py")]}}
    config_hash = digest(config)
    config["config_hash"] = config_hash
    if (outdir / "protocol.json").exists():
        old = json.loads((outdir / "protocol.json").read_text())
        assert old["config_hash"] == config_hash, "Refusing to mix different protocol runs"
    write_json(outdir / "protocol.json", config)
    torch.manual_seed(453)
    torch.cuda.reset_peak_memory_stats()
    runner = Runner(model_info["snapshot"], args.layer - 1, args.max_input)
    print(json.dumps({"phase": "model_loaded", "load_seconds": runner.load_seconds,
                      "layers": len(runner.model.model.layers),
                      "allocated_bytes": torch.cuda.memory_allocated()}), flush=True)
    episodes = make_episodes(args.count, args.dev_count)
    histories = {r["episode_id"]: r for r in read_jsonl(outdir / "histories.jsonl")}
    for e in episodes:
        if e["episode_id"] in histories:
            continue
        h, generations = [], []
        for user in e["users"]:
            h.append({"role": "user", "content": user})
            result = runner.generate([SYSTEM] + h, args.history_tokens)
            generations.append(result)
            h.append({"role": "assistant", "content": result["text"]})
        # No filtering based on either condition's answer or measured gap.
        saved = {**e, "history": h, "history_generation": generations,
                 "history_hash": digest(h), "config_hash": config_hash}
        append_jsonl(outdir / "histories.jsonl", saved)
        histories[e["episode_id"]] = saved
        print(json.dumps({"phase": "history", "episode_id": e["episode_id"],
                          "truncated": sum(x["truncated"] for x in generations)}), flush=True)
    deltas = {}
    audit = []
    for e in episodes:
        d, a = runner.extract(histories[e["episode_id"]]["history"])
        deltas[e["episode_id"]] = d
        audit.append({"episode_id": e["episode_id"], "delta_norm": float(d.norm()), "inputs": a})
    write_json(outdir / "extraction_audit.json", audit)
    dev = torch.stack([deltas[e["episode_id"]] for e in episodes if e["split"] == "dev"])
    mean, basis = build_basis(dev, min(args.rank, args.dev_count))
    projected = {k: basis @ (basis.T @ v) for k, v in deltas.items()}
    torch.save({"mean": mean.cpu(), "basis": basis.cpu(),
                "deltas": {k: v.cpu() for k, v in deltas.items()},
                "config_hash": config_hash}, outdir / "vectors.pt")
    # Verification uses a development history only and an arbitrary fixed query.
    h0 = histories[episodes[0]["episode_id"]]["history"]
    m0 = messages(h0, "Reply with A.", pasted=True)
    l0, _, _ = runner.logits(m0)
    lz, _, _ = runner.logits(m0, deltas[episodes[0]["episode_id"]], 0)
    assert torch.equal(l0, lz), "Alpha-zero invariant failed"
    lm, _, _ = runner.logits(m0, deltas[episodes[0]["episode_id"]], args.alpha)
    write_json(outdir / "hook_check.json", {"alpha_zero_max_error": float((l0 - lz).abs().max()),
                                             "nonzero_logit_change": float((l0 - lm).abs().max())})
    print("extraction and alpha-zero hook check passed", flush=True)
    done_choices = {(x["episode_id"], x["query_id"], x["rotation"], x["method"])
                    for x in read_jsonl(outdir / "choices.jsonl")}
    done_free = {(x["episode_id"], x["method"]) for x in read_jsonl(outdir / "free.jsonl")}
    for i, e in enumerate(episodes):
        eid = e["episode_id"]
        h = histories[eid]["history"]
        # Wrong-history controls stay within a family and use another fixed history.
        peers = [x["episode_id"] for x in episodes if x["family"] == e["family"]]
        wrong_id = peers[(peers.index(eid) + 1) % len(peers)]
        wrong = projected[wrong_id]
        if wrong.norm() > 1e-6:
            wrong = wrong * (projected[eid].norm() / wrong.norm())
        rand_gen = torch.Generator().manual_seed(453 + i)
        random = torch.randn(deltas[eid].shape, generator=rand_gen).to("cuda")
        random *= deltas[eid].norm() / random.norm()
        methods = [("N", None, False, False, 1), ("P", None, True, False, 1),
                   ("Hint", None, True, True, 1), ("D", deltas[eid], True, False, args.alpha),
                   ("G", mean, True, False, args.alpha), ("U", projected[eid], True, False, args.alpha),
                   ("WrongU", wrong, True, False, args.alpha),
                   ("RandomD", random, True, False, args.alpha),
                   ("ReverseD", deltas[eid], True, False, -args.alpha)]
        for q in e["queries"]:
            for rotation in range(3):
                actions = [q["actions"][(j + rotation) % 3] for j in range(3)]
                query = q["text"] + "\nWhich action should you take next?\n" + "\n".join(
                    f"{label}. {act}" for label, act in zip("ABC", actions)) + "\nAnswer only A, B, or C."
                native_probs = native_logp = None
                for method, vector, pasted, hint, alpha in methods:
                    key = (eid, q["query_id"], rotation, method)
                    # N is recomputed to provide the shared current reference, even on resume.
                    if key in done_choices and method != "N":
                        continue
                    msgs = messages(h, query, pasted, hint)
                    labels = runner.label_ids(msgs)
                    logits, sec, length = runner.logits(msgs, vector, alpha)
                    probs_label = logits[labels].softmax(0).cpu().tolist()
                    probs = [0.0] * 3
                    for j, p in enumerate(probs_label):
                        probs[(j + rotation) % 3] = p
                    logp = logits.log_softmax(0)
                    if method == "N":
                        native_probs, native_logp = probs, logp
                    kl = float((native_logp.exp() * (native_logp - logp)).sum())
                    row = dict(episode_id=eid, family=e["family"], split=e["split"],
                               query_id=q["query_id"], rotation=rotation, method=method,
                               action_probs=probs, correct=int(np.argmax(probs)) == q["gold_action"],
                               native_agreement=int(np.argmax(probs)) == int(np.argmax(native_probs)),
                               js_native=js(native_probs, probs), next_token_kl=kl,
                               label_mass=float(logp[labels].exp().sum()),
                               forward_seconds=sec, input_tokens=length, config_hash=config_hash)
                    if key not in done_choices:
                        append_jsonl(outdir / "choices.jsonl", row)
                    done_choices.add(key)
        # One free query per history, balanced across families and update types.
        q = e["queries"][(i // 3) % 3]
        for method, vector, pasted, hint, alpha in methods[:6]:
            key = (eid, method)
            if key in done_free:
                continue
            result = runner.generate(messages(h, q["free_query"], pasted, hint), args.free_tokens, vector, alpha)
            row = dict(episode_id=eid, family=e["family"], split=e["split"], query_id=q["query_id"],
                       method=method, **result, expected_json=q["expected_json"],
                       checker=check_json(result["text"], q["expected_json"]), config_hash=config_hash)
            append_jsonl(outdir / "free.jsonl", row)
            done_free.add(key)
        print(json.dumps({"phase": "evaluated", "episode_id": eid, "split": e["split"]}), flush=True)
    summary = summarize(outdir)
    write_json(outdir / "performance.json", {"model_load_seconds": runner.load_seconds,
                                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                              "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
