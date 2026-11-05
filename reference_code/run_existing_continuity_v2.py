"""Frozen existing MTI triggering/privacy tasks with target-model histories."""
from migration_exploration_core_v1 import *
from run_policy_threeway_v1 import OnlineEngine
from run_policy_practice_v2 import raw_input
import copy,traceback
DATA=ROOT/'data/continuity_dataset_review_2026_10_14';SOURCE=DATA/'candidate_selection_mti_behaviors_v1.json';OUT=ROOT/'results/existing_continuity_v2';STATUS=ROOT/'artifacts/existing_continuity_v2_status.json'
HINT='Answer the current request while honoring all still-active instructions and promises from the conversation. Do not restate information you previously agreed not to mention.'
def assess(text,reference):
    metric=reference['evaluation_metric'];refs=reference['evaluation_reference'];assert metric in ['matching_any','not_matching_any']
    hits=[] if refs is None else [i for i,r in enumerate(refs) if r.lower() in text.lower()]
    value=None if refs is None else float(bool(hits)) if metric=='matching_any' else float(not hits)
    return dict(metric=metric,source_reference_available=refs is not None,score=value,matched_reference_indices=hits,
        caveat='Exact source substring metric, not full semantic reminder execution or privacy protection.')
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);selection=json.loads(SOURCE.read_text());cs=selection['candidates'];assert len(cs)==12 and sum(c['history_user_count'] for c in cs)==27
    cfg=dict(stage='existing_continuity_v2',cases=12,methods=['AOnline','BReplay','CPasted','BSplitNull','BTokenReplay','CStrongHint'],history_cap=1536,output_cap=1536,total_free_cap=99,history_outputs=27,final_outputs=72,
        source_selection='First six triggering and six privacy source-file cases before new inference; no N/P filtering',all_development=True,final_query='Original last user, unmodified; no added cancellation/negative probe',strong_hint=HINT,previous_failed_free_outputs=2,length_repair='v1 stopped after 768-token history truncation; old code/config/attempts preserved; v2 new directory, both caps1536',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,DATA/'mti_all_categories.json',DATA/'mti_metric_functions.py',DATA/'mti_revision.json']},
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['run_existing_continuity_v2.py','run_policy_threeway_v1.py','run_policy_practice_v2.py','migration_exploration_core_v1.py']})
    ch=digest(cfg);write(OUT/'config.json',cfg);write(OUT/'registration.json',dict(config_hash=ch,registered_unix=time.time(),outputs_seen=0));e=OnlineEngine();count=0;hc=0
    for c in cs:
        history=[];consumed=[];cache=None;scores=[]
        for ti,u in enumerate(c['user_turns'][:-1]):
            write(STATUS,dict(phase='constructing_history',case_id=c['case_id'],turn=ti,history_outputs=hc,final_outputs=count))
            canonical=e.serialize(history,u);prefix=e.serialize(history);suffix=canonical[len(prefix):] if history else canonical
            if history:assert canonical[:len(prefix)]==prefix
            actual=consumed+suffix;assert e.tok.decode(actual,skip_special_tokens=False)==e.tok.decode(canonical,skip_special_tokens=False)
            e.calls=[];logits,cache=e.feed(suffix,cache);result,cache,consumed=e.decode(logits,cache,actual,cfg['history_cap']);hc+=1
            append(OUT/'history_attempts.jsonl',dict(case_id=c['case_id'],turn=ti,**result));assert not result['truncated'],'Incomplete history attempt retained'
            history.extend([dict(role='user',content=u),dict(role='assistant',content=result['text'])]);tail=e.tok.encode('<|im_end|>\n',add_special_tokens=False);assert result['generated_ids'][-1]==tail[0]
            _,cache=e.feed(tail,cache);consumed+=tail;canon_prefix=e.serialize(history);assert e.tok.decode(consumed,skip_special_tokens=False)==e.tok.decode(canon_prefix,skip_special_tokens=False)
            s=assess(result['text'],c['scoring_references'][ti]);scores.append(s)
            append(OUT/'history_outputs.jsonl',dict(case_id=c['case_id'],category=c['category'],turn=ti,raw_ids=consumed.copy(),canonical_ids=canon_prefix,retokenization_equal=consumed==canon_prefix,calls=e.calls,score=s,config_hash=ch,**result))
        scored=[s['score'] for s in scores if s['score'] is not None]
        append(OUT/'histories.jsonl',dict(**c,history=history,history_ids=consumed,canonical_history_ids=canon_prefix,history_scores=scores,history_scored_count=len(scored),history_pass_count=sum(scored),history_fully_scored=len(scored)==len(scores),config_hash=ch))
        q=c['final_user_text'];assert q==c['user_turns'][-1]
        for m in cfg['methods']:
            write(STATUS,dict(phase='generating',case_id=c['case_id'],method=m,history_outputs=hc,final_outputs=count))
            actual=q+'\n\n'+HINT if m=='CStrongHint' else q
            inp=raw_input(e,history,consumed,actual) if m in ['AOnline','BTokenReplay'] else e.encode(history,actual,m in ['CPasted','CStrongHint']);e.calls=[]
            if m=='AOnline':cc=copy.deepcopy(cache);logits,cc=e.feed(inp['input_ids'][len(consumed):],cc)
            elif m=='BSplitNull':_,cc=e.feed(canon_prefix);logits,cc=e.feed(inp['input_ids'][len(canon_prefix):],cc)
            else:logits,cc=e.feed(inp['input_ids'])
            first=logits.detach().float().cpu();result,cc,_=e.decode(logits,cc,inp['input_ids'].copy(),cfg['output_cap']);del cc
            append(OUT/'outputs.jsonl',dict(case_id=c['case_id'],category=c['category'],method=m,input=inp,calls=e.calls,score=assess(result['text'],c['scoring_references'][-1]),first_argmax=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.numpy().tobytes()).hexdigest(),config_hash=ch,**result));count+=1
        del cache
    assert count==72 and hc==27;write(OUT/'summary.json',dict(phase='completed',actual_free_outputs=99,history_outputs=hc,final_outputs=count,config_hash=ch));write(STATUS,json.loads((OUT/'summary.json').read_text()))
if __name__=='__main__':
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
