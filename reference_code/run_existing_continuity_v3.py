"""Continue all planned cases, reporting technical history failures, reusing completed case."""
from migration_exploration_core_v1 import *
from run_policy_threeway_v1 import OnlineEngine
from run_policy_practice_v2 import raw_input
from run_existing_continuity_v2 import assess,HINT
import copy,traceback
DATA=ROOT/'data/continuity_dataset_review_2026_10_14';SOURCE=DATA/'candidate_selection_mti_behaviors_v1.json';PRIOR=ROOT/'results/existing_continuity_v2';OUT=ROOT/'results/existing_continuity_v3';STATUS=ROOT/'artifacts/existing_continuity_v3_status.json'
def termination(e,r):
    last=r['generated_ids'][-1]
    if last==e.tok.convert_tokens_to_ids('<|im_end|>'):return 'assistant_eos'
    if last==e.tok.convert_tokens_to_ids('<|im_start|>'):return 'unexpected_role_boundary'
    if last in e.eos:return 'other_eos'
    return 'length_cap'
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);cs=json.loads(SOURCE.read_text())['candidates'];oldh=rows(PRIOR/'histories.jsonl');oldhr=rows(PRIOR/'history_outputs.jsonl');oldr=rows(PRIOR/'outputs.jsonl');assert len(oldh)==1 and len(oldr)==6
    diagnosis=json.loads((ROOT/'artifacts/existing_history_diagnosis_v2.json').read_text());bad=diagnosis['case_id'];assert diagnosis['special_ids_inside']==[151644]
    cfg=dict(stage='existing_continuity_v3',intended_cases=12,known_invalid_history=bad,known_invalid_reason='Unexpected im_start generated inside assistant body, before any final N/P for this case; preserved v2 attempts.',reused_case=oldh[0]['case_id'],reused_free_outputs=9,max_new_free_outputs=81,previous_free_outputs=14,max_actual_across_attempts=95,
        methods=['AOnline','BReplay','CPasted','BSplitNull','BTokenReplay','CStrongHint'],history_cap=1536,output_cap=1536,all_development=True,strong_hint=HINT,
        stop_policy='im_start is a detected unexpected role boundary; do not commit it. Histories with non-assistant EOS, special body tokens or cap are recorded invalid and have no final conditions. Final role/cap stops are retained and flagged, not confirmed migration losses.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,DATA/'mti_all_categories.json',DATA/'mti_metric_functions.py',DATA/'mti_revision.json',PRIOR/'config.json',PRIOR/'histories.jsonl',PRIOR/'history_outputs.jsonl',PRIOR/'outputs.jsonl',PRIOR/'history_attempts.jsonl',ROOT/'artifacts/existing_history_diagnosis_v2.json']},
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['run_existing_continuity_v3.py','run_existing_continuity_v2.py','run_policy_threeway_v1.py','run_policy_practice_v2.py','migration_exploration_core_v1.py']})
    ch=digest(cfg);write(OUT/'config.json',cfg);write(OUT/'registration.json',dict(config_hash=ch,registered_unix=time.time(),new_outputs_seen=0));e=OnlineEngine();e.eos.add(e.tok.convert_tokens_to_ids('<|im_start|>'));count=hc=0
    for c in cs:
        cid=c['case_id']
        if cid==bad:
            append(OUT/'case_status.jsonl',dict(case_id=cid,category=c['category'],status='known_invalid_history',reason=cfg['known_invalid_reason'],diagnosis_source_sha256=cfg['source_sha256']['artifacts/existing_history_diagnosis_v2.json']));continue
        if cid==cfg['reused_case']:
            for r in [x for x in oldhr if x['case_id']==cid]+oldr:
                assert not r['truncated'] and r['generated_ids'][-1]==e.tok.convert_tokens_to_ids('<|im_end|>') and not set(r['token_ids']).intersection(e.tok.all_special_ids)
            h=oldh[0];append(OUT/'histories.jsonl',dict(h,source_config_hash=h['config_hash'],config_hash=ch,reused=True))
            for r in [x for x in oldhr if x['case_id']==cid]:append(OUT/'history_outputs.jsonl',dict(r,source_config_hash=r['config_hash'],config_hash=ch,reused=True,termination='assistant_eos'))
            for r in oldr:append(OUT/'outputs.jsonl',dict(r,source_config_hash=r['config_hash'],config_hash=ch,reused=True,termination='assistant_eos'))
            append(OUT/'case_status.jsonl',dict(case_id=cid,category=c['category'],status='completed_reused',prior='existing_continuity_v2'));continue
        history=[];consumed=[];cache=None;scores=[];invalid=None;canon_prefix=[]
        for ti,u in enumerate(c['user_turns'][:-1]):
            write(STATUS,dict(phase='constructing_history',case_id=cid,turn=ti,new_history_outputs=hc,new_final_outputs=count))
            canonical=e.serialize(history,u);prefix=e.serialize(history);suffix=canonical[len(prefix):] if history else canonical
            if history:assert canonical[:len(prefix)]==prefix
            actual=consumed+suffix;assert e.tok.decode(actual,skip_special_tokens=False)==e.tok.decode(canonical,skip_special_tokens=False)
            e.calls=[];logits,cache=e.feed(suffix,cache);r,cache,consumed=e.decode(logits,cache,actual,cfg['history_cap']);hc+=1;stop=termination(e,r)
            special=sorted(set(r['token_ids']).intersection(e.tok.all_special_ids));append(OUT/'history_attempts.jsonl',dict(case_id=cid,turn=ti,termination=stop,special_body_ids=special,calls=e.calls,config_hash=ch,**r))
            if stop!='assistant_eos' or special:invalid=stop if not special else 'special_body_token';break
            history.extend([dict(role='user',content=u),dict(role='assistant',content=r['text'])]);tail=e.tok.encode('<|im_end|>\n',add_special_tokens=False);_,cache=e.feed(tail,cache);consumed+=tail;canon_prefix=e.serialize(history)
            assert e.tok.decode(consumed,skip_special_tokens=False)==e.tok.decode(canon_prefix,skip_special_tokens=False)
            s=assess(r['text'],c['scoring_references'][ti]);scores.append(s)
            append(OUT/'history_outputs.jsonl',dict(case_id=cid,category=c['category'],turn=ti,raw_ids=consumed.copy(),canonical_ids=canon_prefix,retokenization_equal=consumed==canon_prefix,calls=e.calls,score=s,config_hash=ch,reused=False,termination=stop,**r))
        if invalid:
            del cache;append(OUT/'case_status.jsonl',dict(case_id=cid,category=c['category'],status='invalid_generated_history',turn=ti,reason=invalid));continue
        scored=[s['score'] for s in scores if s['score'] is not None]
        append(OUT/'histories.jsonl',dict(**c,history=history,history_ids=consumed,canonical_history_ids=canon_prefix,history_scores=scores,history_scored_count=len(scored),history_pass_count=sum(scored),history_fully_scored=len(scored)==len(scores),config_hash=ch,reused=False))
        q=c['final_user_text']
        for m in cfg['methods']:
            write(STATUS,dict(phase='generating',case_id=cid,method=m,new_history_outputs=hc,new_final_outputs=count))
            actual=q+'\n\n'+HINT if m=='CStrongHint' else q;inp=raw_input(e,history,consumed,actual) if m in ['AOnline','BTokenReplay'] else e.encode(history,actual,m in ['CPasted','CStrongHint']);e.calls=[]
            if m=='AOnline':cc=copy.deepcopy(cache);logits,cc=e.feed(inp['input_ids'][len(consumed):],cc)
            elif m=='BSplitNull':_,cc=e.feed(canon_prefix);logits,cc=e.feed(inp['input_ids'][len(canon_prefix):],cc)
            else:logits,cc=e.feed(inp['input_ids'])
            first=logits.detach().float().cpu();r,cc,_=e.decode(logits,cc,inp['input_ids'].copy(),cfg['output_cap']);del cc;stop=termination(e,r)
            append(OUT/'outputs.jsonl',dict(case_id=cid,category=c['category'],method=m,input=inp,calls=e.calls,score=assess(r['text'],c['scoring_references'][-1]),first_argmax=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.numpy().tobytes()).hexdigest(),config_hash=ch,reused=False,termination=stop,**r));count+=1
        del cache;append(OUT/'case_status.jsonl',dict(case_id=cid,category=c['category'],status='completed_new'))
    assert hc+count<=81 and len(rows(OUT/'case_status.jsonl'))==12
    write(OUT/'summary.json',dict(phase='completed',new_free_outputs=hc+count,new_history_outputs=hc,new_final_outputs=count,reused_free_outputs=9,previous_free_outputs=14,actual_all_attempts=hc+count+14,valid_histories=len(rows(OUT/'histories.jsonl')),intended_cases=12,config_hash=ch));write(STATUS,json.loads((OUT/'summary.json').read_text()))
if __name__=='__main__':
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
