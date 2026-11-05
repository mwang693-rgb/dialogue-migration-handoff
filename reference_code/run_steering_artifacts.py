"""Exploratory steering on frozen artifact histories; no blind test or tuning claim."""
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import torch
from artifact_protocol_v2 import check
from run_pilot import Runner, append_jsonl, build_basis, digest, intervention, messages, read_jsonl, write_json

OUT = Path("results/steering_artifacts_v1")
SOURCE = Path("results/artifact_protocol_v2")
CUE = "Acknowledge that this conversation history is available. Reply briefly."
POSITIONS = [0, 1, 3, 7, 15, 31, 63, 95]


@torch.inference_mode()
def extract(runner, history):
    states, audits = [], []
    for pasted in [False, True]:
        msgs = messages(history, CUE, pasted=pasted)
        rendered, ids = runner.tokenize(msgs)
        captured = []
        def capture(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[0, -1].float().clone())
        handle = runner.layer.register_forward_hook(capture)
        try:
            runner.logits(msgs)
        finally:
            handle.remove()
        assert len(captured) == 1
        states.append(captured[0])
        audits.append(dict(format="P" if pasted else "N", rendered=rendered, token_ids=ids,
            fixed_cue=CUE, final_task_query_included=False, terminal_token_id=ids[-1]))
    assert audits[0]["terminal_token_id"] == audits[1]["terminal_token_id"]
    return states[0] - states[1], audits


@torch.inference_mode()
def prefix_distributions(runner, msgs, ref_ids, vector=None, alpha=1.0):
    _, ids = runner.tokenize(msgs)
    selected = [i for i in POSITIONS if i < len(ref_ids)]
    assert selected
    inp = torch.tensor([ids], device="cuda")
    cache = None
    logs = []
    with intervention(runner.layer, vector, alpha):
        for pos in range(max(selected) + 1):
            mask = torch.ones((1, len(ids) + pos), device="cuda", dtype=torch.long)
            out = runner.model(input_ids=inp, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            if pos in selected:
                logs.append(out.logits[0, -1].float().log_softmax(0).clone())
            if pos < max(selected):
                inp = torch.tensor([[ref_ids[pos]]], device="cuda")
    return selected, torch.stack(logs)


def divergences(native, other, positions):
    logm = torch.logaddexp(native, other) - torch.log(torch.tensor(2., device=native.device))
    kl = (native.exp() * (native - other)).sum(-1)
    js = ((native.exp() * (native - logm)).sum(-1) + (other.exp() * (other - logm)).sum(-1)) / 2
    return dict(positions=positions, kl_by_position=kl.tolist(), js_by_position=js.tolist(),
        mean_reference_prefix_kl=float(kl.mean()), mean_reference_prefix_js=float(js.mean()),
        top1_agreement=float((native.argmax(-1) == other.argmax(-1)).float().mean()))


@torch.inference_mode()
def native_with_ids(runner, msgs):
    _, ids = runner.tokenize(msgs)
    inp = torch.tensor([ids], device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = runner.model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=False,
        max_new_tokens=1024, use_cache=True, pad_token_id=runner.tokenizer.eos_token_id, logits_to_keep=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gen = out[0, len(ids):]
    eos = runner.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    natural = len(gen) > 0 and int(gen[-1]) in eos
    reference = gen.tolist()
    if natural:
        reference = reference[:-1]
    result = dict(text=runner.tokenizer.decode(gen, skip_special_tokens=True), generated_tokens=len(gen),
        input_tokens=len(ids), truncated=not natural and len(gen) >= 1024, seconds=elapsed)
    return result, reference[:96]


def summarize(rows, histories):
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    nby = {(r["case_id"], r["query_id"]):r for r in rows if r["method"] == "N"}
    summaries = {}
    for method, rs in by.items():
        summaries[method] = dict(outputs=len(rs), passed=sum(r["checker"]["all_pass"] for r in rs),
            capped=sum(r["truncated"] for r in rs), exact_native_text=sum(r["text"] == nby[(r["case_id"],r["query_id"])]["text"] for r in rs),
            mean_reference_prefix_kl=sum(r["distribution"]["mean_reference_prefix_kl"] for r in rs) / len(rs),
            mean_reference_prefix_js=sum(r["distribution"]["mean_reference_prefix_js"] for r in rs) / len(rs))
        summaries[method]["families"] = {f:dict(outputs=sum(r["family"] == f for r in rs), passed=sum(r["family"] == f and r["checker"]["all_pass"] for r in rs)) for f in ["book", "code", "table"]}
    return dict(phase="completed", histories=len(histories), paired_queries=len(nby), methods=summaries,
        status="exploratory development data, not independent held-out test; single fixed layer/strength, no optimization yet")


def main():
    assert not OUT.exists(), "Keep earlier runs; repairs require a new directory"
    summary = json.loads((SOURCE / "summary.json").read_text())
    assert summary["constructed"] == 12 and summary["outputs"] == 48 and not summary["errors"]
    hs = read_jsonl(SOURCE / "histories.jsonl")
    assert len(hs) == 12 and all(not h["history_truncated"] for h in hs)
    fit_ids = [h["case_id"] for family in ["book", "code", "table"] for h in [x for x in hs if x["family"] == family][:2]]
    config = dict(model="Qwen2.5-7B-Instruct", dtype="bfloat16", seed=453, layer_one_based=16, alpha=1.0, rank=3,
        do_sample=False, repetition_penalty=1.0, sampling_filters=None,
        answer_cap=1024, max_input=8192, positions=POSITIONS, reference_cap=96, fixed_extraction_cue=CUE,
        extraction="paired N/P history plus fixed task-independent cue, final assistant generation boundary",
        intervention="one post-block residual; final prefill token plus every subsequent cached decoding token",
        prefix_evaluation="same interventions during cached teacher forcing; identical exact generated native token IDs; EOS excluded; raw model conditional distributions, not greedy decoder point masses",
        previous_decoder="data construction and old outputs used Qwen default repetition_penalty=1.05; all N/P free outputs regenerated here at 1.0, not mixed",
        fit_source_ids=fit_ids, fit_role="six histories fit G/U; all twelve are previously explored development examples, not blind test",
        methods=["N", "P", "Hint", "D", "G", "U", "WrongD", "RandomD", "ReverseD"],
        alpha_zero="exact logit equality checked on every query; redundant free generation omitted",
        code_sha256={n:hashlib.sha256(Path("scripts", n).read_bytes()).hexdigest() for n in ["run_steering_artifacts.py", "artifact_protocol_v2.py", "run_artifact_expansion.py", "run_pilot.py", "pilot_data.py"]},
        source_sha256={n:hashlib.sha256((SOURCE / n).read_bytes()).hexdigest() for n in ["config.json", "histories.jsonl", "outputs.jsonl"]})
    OUT.mkdir(parents=True)
    write_json(OUT / "config.json", config)
    ch = digest(config)
    torch.manual_seed(453)
    runner = Runner("/root/autodl-tmp/models/qwen2.5-7b-instruct", 15, 8192)
    assert runner.model.generation_config.repetition_penalty == 1.05
    runner.model.generation_config.repetition_penalty = 1.0
    write_json(OUT / "effective_generation_config.json", runner.model.generation_config.to_dict())
    ds = {}
    for h in hs:
        cid = h["case_id"]
        delta, audit = extract(runner, h["history"])
        assert torch.isfinite(delta).all() and delta.norm() > 0
        ds[cid] = delta
        append_jsonl(OUT / "extraction_audit.jsonl", dict(case_id=cid, config_hash=ch, delta_norm=float(delta.norm()), audit=audit))
    mean, basis = build_basis(torch.stack([ds[cid] for cid in fit_ids]), 3)
    us = {cid:basis @ (basis.T @ delta) for cid,delta in ds.items()}
    torch.save(dict(deltas={k:v.cpu() for k,v in ds.items()}, mean=mean.cpu(), basis=basis.cpu(), projected={k:v.cpu() for k,v in us.items()}, config_hash=ch), OUT / "vectors.pt")
    generator = torch.Generator(device="cpu").manual_seed(453)
    ids = [h["case_id"] for h in hs]
    for index, h in enumerate(hs):
        cid, family = h["case_id"], h["family"]
        wrong_id = ids[(index + 4) % len(ids)]
        wrong = ds[wrong_id] * (ds[cid].norm() / ds[wrong_id].norm())
        random = torch.randn(ds[cid].shape, generator=generator).to("cuda")
        random *= ds[cid].norm() / random.norm()
        methods = [("N", None, False, False, 1.), ("P", None, True, False, 1.), ("Hint", None, True, True, 1.),
            ("D", ds[cid], True, False, 1.), ("G", mean, True, False, 1.), ("U", us[cid], True, False, 1.),
            ("WrongD", wrong, True, False, 1.), ("RandomD", random, True, False, 1.), ("ReverseD", ds[cid], True, False, -1.)]
        for q in h["queries"]:
            nmsg = messages(h["history"], q["text"])
            native_result, ref_ids = native_with_ids(runner, nmsg)
            assert ref_ids
            positions, nlogs = prefix_distributions(runner, nmsg, ref_ids)
            if index == 0 and q == h["queries"][0]:
                direct, _, _ = runner.logits(nmsg)
                err = float((direct.log_softmax(0) - nlogs[0]).abs().max())
                assert err < 1e-5, f"Cached reference first-position mismatch: {err}"
                write_json(OUT / "prefix_forward_check.json", dict(first_position_max_logprob_error=err, passed=True))
            pmsg = messages(h["history"], q["text"], pasted=True)
            p0, _, _ = runner.logits(pmsg)
            pz, _, _ = runner.logits(pmsg, ds[cid], 0.)
            assert torch.equal(p0, pz), "alpha=0 changed logits"
            append_jsonl(OUT / "references.jsonl", dict(case_id=cid, query_id=q["query_id"], config_hash=ch,
                reference_ids=ref_ids, reference_origin="exact native generated token IDs, same prefix across conditions", alpha_zero_exact=True))
            for method, vector, pasted, hint, alpha in methods:
                print("STEERING", cid, q["query_id"], method, flush=True)
                write_json(Path("artifacts/steering_artifacts_status.json"), dict(phase="running", case_id=cid, query_id=q["query_id"], method=method, completed_histories=index, total_histories=12, updated_unix=time.time()))
                msg = messages(h["history"], q["text"], pasted=pasted, hint=hint)
                start = time.perf_counter()
                if method == "N":
                    dist = divergences(nlogs, nlogs, positions)
                else:
                    pos, logs = prefix_distributions(runner, msg, ref_ids, vector, alpha)
                    assert pos == positions
                    dist = divergences(nlogs, logs, positions)
                distribution_seconds = time.perf_counter() - start
                if method == "N":
                    result = native_result
                    origin = "new native generation at repetition_penalty=1.0"
                else:
                    result = runner.generate(msg, 1024, vector, alpha)
                    origin = "new free generation"
                append_jsonl(OUT / "outputs.jsonl", dict(case_id=cid, query_id=q["query_id"], family=family,
                    method=method, config_hash=ch, origin=origin, checker=check(result["text"], family, q),
                    distribution=dist, distribution_seconds=distribution_seconds,
                    vector_norm=float(vector.norm()) if vector is not None else 0., alpha=alpha,
                    wrong_history_id=wrong_id if method == "WrongD" else None, **result))
    rows = read_jsonl(OUT / "outputs.jsonl")
    keys = [(r["case_id"],r["query_id"],r["method"]) for r in rows]
    expected = {(h["case_id"],q["query_id"],m) for h in hs for q in h["queries"] for m in config["methods"]}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert all(r["config_hash"] == ch for r in rows)
    result = summarize(rows, hs)
    result["config_hash"] = ch
    result["outputs"] = len(rows)
    write_json(OUT / "summary.json", result)
    write_json(Path("artifacts/steering_artifacts_status.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
