"""Four-condition generation; nat/flat activation shards for the CPU probe."""
import argparse
import copy
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, rows, sha, write, append, validate_cases, check_rule, request_text


def capture_feed(engine, ids, cache, layers):
    import torch
    states, handles = {}, []
    def capture(index):
        def hook(module, args, output):
            value = output[0] if isinstance(output,tuple) else output
            states[index] = value[0,-1].detach().float().cpu().clone()
        return hook
    for layer in layers:
        module = engine.model.model.embed_tokens if layer == 0 else engine.model.model.layers[layer-1]
        handles.append(module.register_forward_hook(capture(layer)))
    try:
        logits, new_cache = engine.feed(ids,cache)
    finally:
        for h in handles: h.remove()
    assert set(states)==set(layers)
    return logits,new_cache,states


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--registration',type=Path)
    p.add_argument('--development-smoke',action='store_true',help='At most two declared old development cases; never counted as confirmation')
    p.add_argument('--protocol',type=Path,default=ROOT/'confirmation/protocol.json')
    p.add_argument('--model',type=Path,required=True,help='Local pinned model snapshot')
    p.add_argument('--snapshot-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert not a.out.exists(), 'Use a new run directory; no overwrite or implicit resume'
    cfg,cases=read(a.protocol),rows(a.cases)
    if a.development_smoke:
        assert len(cases)<=2 and cases and all(c.get('development_fixture') is True for c in cases)
        reg=dict(status='development_smoke_not_confirmation',protocol_sha256=sha(a.protocol),cases_sha256=sha(a.cases))
    else:
        assert a.registration, 'A real confirmation run requires a frozen registration'
        reg=read(a.registration)
        assert cfg['status']=='reviewed' and reg['status']=='registered_before_generation'
        assert sha(a.protocol)==reg['protocol_sha256'] and sha(a.cases)==reg['cases_sha256']
        for f,d in reg['code_sha256'].items(): assert sha(ROOT/f)==d, f
        assert not any(c.get('development_fixture') for c in cases)
    validate_cases(cases,cfg,confirmation=not a.development_smoke)
    run_kind='development_smoke' if a.development_smoke else 'confirmation'
    assert a.model.is_dir()
    snapshot=read(a.snapshot_manifest)
    assert snapshot.get('all_weights_sha256_verified') and snapshot['revision']==cfg['revision']
    assert Path(snapshot['snapshot']).resolve()==a.model.resolve()
    a.out.mkdir(parents=True)
    (a.out/'activations').mkdir()
    write(a.out/'protocol.json',cfg);write(a.out/'registration.json',reg)
    write(a.out/'status.json',dict(phase='loading_model',run_kind=run_kind,started_utc=datetime.now(timezone.utc).isoformat()))
    (a.out/'cases.jsonl').write_bytes(a.cases.read_bytes())
    sys.path.insert(0,str(ROOT/'reference_code'))
    import torch
    import migration_exploration_core_v1 as core
    core.MODEL=str(a.model.resolve())
    from run_policy_threeway_v1 import OnlineEngine
    start=time.perf_counter()
    try:
        e=OnlineEngine();e.eos.add(e.tok.convert_tokens_to_ids('<|im_start|>'))
        assert max(cfg['layers'])==e.model.config.num_hidden_layers
        write(a.out/'model_config.json',e.model.config.to_dict())
        model_files=sorted(a.model.glob('*.safetensors'))+sorted(a.model.glob('*.json'))
        write(a.out/'environment.json',dict(torch=torch.__version__, model_path=str(a.model), model_revision_expected=cfg['revision'], tokenizer_class=type(e.tok).__name__, device=torch.cuda.get_device_name(0), hidden_size=e.model.config.hidden_size, local_model_sha256={f.name:sha(f) for f in model_files}, identity_note='Local files are hashed before generation. Review against the intended pinned snapshot; a directory name alone is not model verification.'))
        accepted={name:0 for name in cfg['rules']}
        for c in cases:
            name=c['inst_name']
            if not a.development_smoke and accepted[name]>=cfg['target_per_rule']:
                append(a.out/'case_status.jsonl',dict(case_id=c['case_id'],source_id=c['source_id'],status='unused_predeclared_reserve'))
                continue
            write(a.out/'status.json',dict(phase='generating_history',run_kind=run_kind,case_id=c['case_id'],accepted_by_rule=accepted,seconds=time.perf_counter()-start))
            history,consumed,cache=[],[],None
            scores=[]
            for i,q in enumerate(c['history_questions']):
                u=(c['rule_text']+'\n\n' if i==0 else '')+q
                canonical=e.serialize(history);full=e.serialize(history,u)
                if history: assert consumed==canonical and full[:len(canonical)]==canonical
                suffix=full[len(canonical):] if history else full
                actual=consumed+suffix;e.calls=[]
                logits,cache=e.feed(suffix,cache)
                r,cache,consumed=e.decode(logits,cache,actual,cfg['cap'])
                append(a.out/'history_attempts.jsonl',dict(case_id=c['case_id'],source_id=c['source_id'],turn=i,calls=e.calls,**r))
                if r['truncated'] or r['generated_ids'][-1]!=e.tok.convert_tokens_to_ids('<|im_end|>') or set(r['token_ids']).intersection(e.tok.all_special_ids):
                    append(a.out/'case_status.jsonl',dict(case_id=c['case_id'],source_id=c['source_id'],status='invalid_history',turn=i))
                    break
                scores.append(check_rule(r['text'],'ordinary',False,cfg['word_limit'],name,c['inst_args']))
                if not scores[-1]['passed']:
                    append(a.out/'case_status.jsonl',dict(case_id=c['case_id'],source_id=c['source_id'],status='rule_not_established',turn=i,score=scores[-1]))
                    break
                history += [dict(role='user',content=u),dict(role='assistant',content=r['text'])]
                tail=e.tok.encode('<|im_end|>\n',add_special_tokens=False)
                _,cache=e.feed(tail,cache);consumed+=tail
                assert consumed==e.serialize(history), 'Retokenization drift: preserve run and inspect'
            else:
                accepted[name]+=1
                append(a.out/'histories.jsonl',dict(**c,history=history,history_ids=consumed,history_scores=scores,established=True))
                vectors=[]
                for request_type in ['ordinary','change']:
                    native_ids=None
                    for condition in cfg['conditions']:
                        write(a.out/'status.json',dict(phase='generating_test',run_kind=run_kind,case_id=c['case_id'],request_type=request_type,condition=condition,accepted_by_rule=accepted,seconds=time.perf_counter()-start))
                        actual_q=request_text(c['question'],request_type,condition,cfg,c)
                        inp=e.encode(history,actual_q,pasted=condition in ['C','C_reminder'])
                        ids=inp['input_ids'];e.calls=[]
                        assert len(ids)<=cfg['max_input_tokens'], 'Registered input cap exceeded'
                        if condition=='A': native_ids=ids.copy()
                        if condition=='B': assert native_ids==ids, 'Online and role replay prompt IDs differ'
                        cc=copy.deepcopy(cache) if condition=='A' else None
                        suffix=ids[len(consumed):] if condition=='A' else ids
                        if condition=='A': assert ids[:len(consumed)]==consumed
                        if condition in ['B','C']:
                            logits,cc,states=capture_feed(e,suffix,cc,cfg['layers'])
                            for layer,vector in states.items():
                                vectors.append(dict(source_id=c['source_id'],case_id=c['case_id'],query_id=request_type,request_type=request_type,condition='nat' if condition=='B' else 'flat',run_condition=condition,layer=layer,position=len(ids)-1,n_tokens=len(ids),terminal_token_id=ids[-1],vector=vector))
                        else:
                            logits,cc=e.feed(suffix,cc)
                        r,cc,_=e.decode(logits,cc,ids.copy(),cfg['cap'])
                        termination='cap' if r['truncated'] else 'assistant_eos' if r['generated_ids'][-1]==e.tok.convert_tokens_to_ids('<|im_end|>') else 'unexpected_role_stop'
                        method={'A':'AOnline','B':'BReplay','C':'CPasted','C_reminder':'CSpecificRule'}[condition]
                        append(a.out/'outputs.jsonl',dict(run_kind=run_kind,source_id=c['source_id'],case_id=c['case_id'],inst_name=name,inst_args=c['inst_args'],query=request_type,query_id=request_type,request_type=request_type,condition=condition,method=method,query_text=actual_q,input_ids=ids,calls=e.calls,termination=termination,score=check_rule(r['text'],request_type,termination!='assistant_eos',cfg['word_limit'],name,c['inst_args']),**r))
                        del cc
                assert len(vectors)==len(cfg['layers'])*4
                for row in vectors: assert row['vector'].shape==(e.model.config.hidden_size,) and torch.isfinite(row['vector']).all()
                for typ in ['ordinary','change']:
                    pair=[r for r in vectors if r['request_type']==typ and r['layer']==0]
                    assert pair[0]['terminal_token_id']==pair[1]['terminal_token_id']
                    assert torch.equal(pair[0]['vector'],pair[1]['vector']), 'Token-embedding control mismatch'
                torch.save(dict(schema_version=1,is_synthetic=False,run_kind=run_kind,hidden_size=e.model.config.hidden_size,records=vectors),a.out/'activations'/f"{c['case_id']}.pt")
                append(a.out/'case_status.jsonl',dict(case_id=c['case_id'],source_id=c['source_id'],status='completed',established=all(s['passed'] for s in scores)))
            del cache
        enough=a.development_smoke or all(v==cfg['target_per_rule'] for v in accepted.values())
        write(a.out/'status.json',dict(phase='completed' if enough else 'insufficient_established_histories',run_kind=run_kind,accepted_by_rule=accepted,seconds=time.perf_counter()-start,activation_files=len(list((a.out/'activations').glob('*.pt'))),steering_run=False))
        assert enough, 'Predeclared reserve exhausted; do not add post-outcome cases'
    except Exception:
        write(a.out/'status.json',dict(phase='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
