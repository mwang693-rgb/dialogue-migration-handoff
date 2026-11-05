"""New frozen question streams, genuine online history and cue factorial controls."""
from style_confirmation_common_v2 import *
from migration_exploration_core_v1 import Engine,digest,append
from run_policy_threeway_v1 import OnlineEngine
from run_policy_practice_v2 import raw_input
from run_existing_continuity_v3 import termination
import time,sys

STATUS=ROOT/'artifacts/style_confirmation_v2_status.json'
PROTOCOL=ROOT/'新题流确认与单历史调节协议_v2_2026-10-14.md'

def preflight():
    selection=read(DATA/'selection.json');cases=read(DATA/'cases.json')
    assert len(cases)==4 and len({c['source_id'] for c in cases})==4
    assert sha(ROOT/'scripts/build_style_confirmation_v2.py')==selection['code_sha256']
    assert sha(ROOT/'data/continuity_dataset_review_2026_10_14/mteval_global_multiturn.jsonl')==selection['public_source_sha256']
    for item in selection['prior_histories']:assert sha(ROOT/item['path'])==item['sha256']
    e=Engine(load=False);inputs=[]
    for c in cases:
        # CPU-only placeholder histories test exact role/template encoding before model load.
        h=[]
        for u in c['history_users']:h.extend([dict(role='user',content=u),dict(role='assistant',content='Acknowledged.')])
        for qname,q in c['queries'].items():
            ms=selection['normal_methods'] if qname!='override_q1' else selection['override_methods']
            for m in ms:
                altered=variant(h,c['style_clause'],m.endswith('Remove'))
                inp=e.encode(altered,q,m.startswith('C'));inputs.append(len(inp['input_ids']))
    assert len(inputs)==52
    return dict(verified=True,cases=4,source_questions=20,prior_history_files_checked=len(selection['prior_histories']),placeholder_inputs=52,max_placeholder_tokens=max(inputs),new_generation=0)

def main():
    checks=preflight()
    if '--check-only' in sys.argv:
        write(ROOT/'artifacts/style_confirmation_preflight_v2.json',checks);print(json.dumps(checks));return
    assert not CONF.exists();CONF.mkdir(parents=True)
    selection=read(DATA/'selection.json');cases=read(DATA/'cases.json')
    cfg=dict(stage='style_confirmation_v2',selection=selection,cap=1536,max_free_outputs=64,
             extra_gate='At least three of four committed histories have at least one listed short form; no per-case outcome filtering.',
             planned_steering=dict(layer=LAYER,rho=RHO,methods=STEER_METHODS,queries=['q0','override_q1'],max_free_outputs=48),
             protocol_sha256=sha(PROTOCOL),
             source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'cases.json',DATA/'selection.json',DATA/'selection_decisions.json']},
             code_sha256={n:sha(ROOT/'scripts'/n) for n in ['run_style_confirmation_v2.py','style_confirmation_common_v2.py','style_cue_verification_common_v1.py','migration_exploration_core_v1.py','run_policy_threeway_v1.py','run_policy_practice_v2.py','run_existing_continuity_v3.py']})
    ch=digest(cfg);write(CONF/'config.json',cfg);write(CONF/'registration.json',dict(config_hash=ch,outputs_seen=0,registered_unix=time.time()));write(CONF/'preflight.json',checks)
    e=OnlineEngine();e.eos.add(e.tok.convert_tokens_to_ids('<|im_start|>'));hc=fc=0
    for c in cases:
        history=[];consumed=[];cache=None;invalid=None
        for t,u in enumerate(c['history_users']):
            write(STATUS,dict(phase='history',case_id=c['case_id'],turn=t,history_outputs=hc,final_outputs=fc))
            full=e.serialize(history,u);prefix=e.serialize(history);suffix=full[len(prefix):] if history else full
            assert not history or full[:len(prefix)]==prefix
            actual=consumed+suffix;assert e.tok.decode(actual,skip_special_tokens=False)==e.tok.decode(full,skip_special_tokens=False)
            e.calls=[];logits,cache=e.feed(suffix,cache);r,cache,consumed=e.decode(logits,cache,actual.copy(),1536);hc+=1
            stop=termination(e,r);special=sorted(set(r['token_ids']).intersection(e.tok.all_special_ids))
            append(CONF/'history_attempts.jsonl',dict(case_id=c['case_id'],turn=t,calls=e.calls.copy(),termination=stop,special_body_ids=special,config_hash=ch,**r))
            if stop!='assistant_eos' or special:invalid=stop if not special else 'special_body_token';break
            history.extend([dict(role='user',content=u),dict(role='assistant',content=r['text'])])
            tail=e.tok.encode('<|im_end|>\n',add_special_tokens=False);_,cache=e.feed(tail,cache);consumed+=tail
            canonical=e.serialize(history)
            assert e.tok.decode(consumed,skip_special_tokens=False)==e.tok.decode(canonical,skip_special_tokens=False)
            append(CONF/'history_outputs.jsonl',dict(case_id=c['case_id'],turn=t,raw_ids=consumed.copy(),canonical_ids=canonical,calls=e.calls.copy(),score=assess(r['text']),config_hash=ch,**r))
        if invalid:
            append(CONF/'case_status.jsonl',dict(case_id=c['case_id'],status='invalid_history',turn=t,reason=invalid));del cache;continue
        append(CONF/'histories.jsonl',dict(**c,history=history,history_ids=consumed.copy(),canonical_history_ids=canonical,config_hash=ch))
        for qname,q in c['queries'].items():
            methods=selection['normal_methods'] if qname!='override_q1' else selection['override_methods']
            for m in methods:
                write(STATUS,dict(phase='final',case_id=c['case_id'],query=qname,method=m,history_outputs=hc,final_outputs=fc))
                altered=variant(history,c['style_clause'],m.endswith('Remove'))
                inp=raw_input(e,history,consumed,q) if m=='AOnline' else e.encode(altered,q,m.startswith('C'))
                e.calls=[]
                if m=='AOnline':cc=copy.deepcopy(cache);logits,cc=e.feed(inp['input_ids'][len(consumed):],cc)
                else:logits,cc=e.feed(inp['input_ids'])
                first=int(logits.argmax());r,cc,_=e.decode(logits,cc,inp['input_ids'].copy(),1536);del cc;fc+=1
                append(CONF/'outputs.jsonl',dict(case_id=c['case_id'],source_id=c['source_id'],flow=c['flow'],query=qname,method=m,input=inp,calls=e.calls.copy(),first_argmax=first,score=assess(r['text']),termination=termination(e,r),config_hash=ch,**r))
        del cache;append(CONF/'case_status.jsonl',dict(case_id=c['case_id'],status='completed'))
    assert hc+fc<=64
    result=analyze_confirmation();write(CONF/'analysis.json',result)
    summary=dict(phase='completed',history_attempts=hc,final_outputs=fc,total_free_outputs=hc+fc,gate_pass=result['gate_pass'],gate_reasons=result['gate_reasons'],config_hash=ch)
    write(CONF/'summary.json',summary);write(STATUS,summary);print(json.dumps(summary))

if __name__=='__main__':
    import traceback
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
