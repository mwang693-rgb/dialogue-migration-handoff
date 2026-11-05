"""Audit sparse-reference KL at every token and at the first free-generation divergence."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import torch
from run_pilot import Runner, append_jsonl, digest, intervention, messages, read_jsonl, write_json

OUT = Path("results/dense_reference_audit_v1")
SOURCE = Path("results/steering_artifacts_v1")


@torch.inference_mode()
def generated_reference(runner, msgs, expected):
    _, ids = runner.tokenize(msgs)
    inp = torch.tensor([ids],device="cuda")
    seq = runner.model.generate(input_ids=inp,attention_mask=torch.ones_like(inp),do_sample=False,
        max_new_tokens=1024,use_cache=True,pad_token_id=runner.tokenizer.eos_token_id,logits_to_keep=1)
    gen = seq[0,len(ids):].tolist()
    text = runner.tokenizer.decode(gen,skip_special_tokens=True)
    assert text == expected, "Generation changed; use a new protocol instead of mixing references"
    eos = runner.model.generation_config.eos_token_id
    eos = eos if isinstance(eos,list) else [eos]
    if gen and gen[-1] in eos:
        gen = gen[:-1]
    return gen


@torch.inference_mode()
def dense_logs(runner, msgs, ref, vector=None, alpha=1.):
    _, ids = runner.tokenize(msgs)
    inp = torch.tensor([ids],device="cuda")
    cache,logs = None,[]
    with intervention(runner.layer,vector,alpha):
        for pos in range(len(ref)):
            out = runner.model(input_ids=inp,attention_mask=torch.ones((1,len(ids)+pos),device="cuda",dtype=torch.long),
                past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache = out.past_key_values
            logs.append(out.logits[0,-1].double().log_softmax(0).clone())
            inp = torch.tensor([[ref[pos]]],device="cuda")
    return torch.stack(logs)


def compare(nlogs,other,ref):
    logm = torch.logaddexp(nlogs,other)-torch.log(torch.tensor(2.,device=nlogs.device,dtype=torch.float64))
    kl = (nlogs.exp()*(nlogs-other)).sum(-1)
    js = ((nlogs.exp()*(nlogs-logm)).sum(-1)+(other.exp()*(other-logm)).sum(-1))/2
    indices = torch.tensor(ref,device=nlogs.device).view(-1,1)
    return dict(dense_mean_kl=float(kl.mean()),dense_mean_js=float(js.mean()),
        kl_by_position=kl.tolist(),js_by_position=js.tolist(),
        native_ref_token_logprobs=nlogs.gather(1,indices).view(-1).tolist(),
        condition_ref_token_logprobs=other.gather(1,indices).view(-1).tolist(),
        native_top1=nlogs.argmax(-1).tolist(),condition_top1=other.argmax(-1).tolist())


def main():
    assert not OUT.exists(), "Use new directories for repairs"
    hs = read_jsonl(Path("results/artifact_protocol_v2/histories.jsonl"))
    hs = [h for h in hs if h["case_id"] in ["artifact-book-0","artifact-code-0","artifact-table-0"]]
    assert len(hs) == 3
    original = {(r["case_id"],r["query_id"],r["method"]):r for r in read_jsonl(SOURCE/"outputs.jsonl")}
    config = dict(model="Qwen2.5-7B-Instruct",dtype="bfloat16",repetition_penalty=1.0,seed=453,layer_one_based=16,
        selected_histories=[h["case_id"] for h in hs],reference_cap=384,reference_paths=["N","P"],
        methods=["N","P","Hint","D","ReverseD"],expected_rows=60,
        purpose="metric coverage/numerical audit on previously explored tasks, not method selection or test",
        log_softmax_precision="float64 from BF16 model logits",no_new_free_method_outputs=True,
        code_sha256={n:hashlib.sha256(Path("scripts",n).read_bytes()).hexdigest() for n in ["audit_dense_reference.py","run_pilot.py","pilot_data.py"]},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/"outputs.jsonl",SOURCE/"vectors.pt",Path("results/artifact_protocol_v2/histories.jsonl")]})
    OUT.mkdir(parents=True)
    write_json(OUT/"config.json",config)
    ch = digest(config)
    torch.manual_seed(453)
    runner = Runner("/root/autodl-tmp/models/qwen2.5-7b-instruct",15,8192)
    runner.model.generation_config.repetition_penalty = 1.0
    vectors = torch.load(SOURCE/"vectors.pt",map_location="cpu",weights_only=True)["deltas"]
    done = 0
    for h in hs:
        cid = h["case_id"]
        delta = vectors[cid].to("cuda")
        for q in h["queries"]:
            texts = {m:original[(cid,q["query_id"],m)]["text"] for m in ["N","P"]}
            exact_refs = {m:generated_reference(runner,messages(h["history"],q["text"],pasted=m=="P"),texts[m]) for m in ["N","P"]}
            first_diff = next((i for i,(x,y) in enumerate(zip(exact_refs["N"],exact_refs["P"])) if x != y),None)
            if first_diff is None and len(exact_refs["N"]) != len(exact_refs["P"]):
                first_diff = min(len(exact_refs["N"]),len(exact_refs["P"]))
            for ref_method in ["N","P"]:
                ref = exact_refs[ref_method][:384]
                nlogs = dense_logs(runner,messages(h["history"],q["text"]),ref)
                for method in config["methods"]:
                    print("DENSE",done,cid,q["query_id"],ref_method,method,flush=True)
                    write_json(Path("artifacts/dense_reference_status.json"),dict(phase="running",completed=done,total=60,case_id=cid,query_id=q["query_id"],reference=ref_method,method=method,updated_unix=time.time()))
                    if method == "N":
                        other = nlogs
                    else:
                        vector = delta if method in ["D","ReverseD"] else None
                        other = dense_logs(runner,messages(h["history"],q["text"],pasted=True,hint=method=="Hint"),ref,
                            vector,-1. if method == "ReverseD" else 1.)
                    stats = compare(nlogs,other,ref)
                    sparse = [i for i in [0,1,3,7,15,31,63,95] if i < len(ref)]
                    stats["old_sparse_mean_recomputed64"] = sum(stats["kl_by_position"][i] for i in sparse)/len(sparse)
                    stats["first_native_paste_free_divergence"] = first_diff
                    stats["kl_at_first_free_divergence"] = stats["kl_by_position"][first_diff] if first_diff is not None and first_diff < len(ref) else None
                    append_jsonl(OUT/"distributions.jsonl",dict(case_id=cid,query_id=q["query_id"],reference=ref_method,
                        method=method,config_hash=ch,reference_ids=ref,natural_reference_length=len(exact_refs[ref_method]),
                        reference_capped=len(exact_refs[ref_method])>384,**stats))
                    done += 1
                    if method != "N":
                        del other
                del nlogs
    rows = read_jsonl(OUT/"distributions.jsonl")
    assert len(rows) == 60 and len({(r["case_id"],r["query_id"],r["reference"],r["method"]) for r in rows}) == 60
    summary = dict(phase="completed",rows=60,config_hash=ch,reference_capped=sum(r["reference_capped"] for r in rows),
        metrics=[{k:r[k] for k in ["case_id","query_id","reference","method","dense_mean_kl","old_sparse_mean_recomputed64","first_native_paste_free_divergence","kl_at_first_free_divergence"]} for r in rows])
    write_json(OUT/"summary.json",summary)
    write_json(Path("artifacts/dense_reference_status.json"),summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
