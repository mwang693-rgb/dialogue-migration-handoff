"""Preserve actual generated history tokens; expose text-retokenization controls."""
from migration_exploration_core_v1 import *
from run_policy_threeway_v1 import OnlineEngine
import copy,traceback
DATA=ROOT/'data/policy_practice_v1';OUT=ROOT/'results/policy_practice_v2';STATUS=ROOT/'artifacts/policy_practice_v2_status.json'
def assess(text,policy,gold,release):
    ids=re.findall(r'ITEM_[A-H]',text)
    if release:s=len(ids)==len(gold) and len(set(ids))==len(ids);content=ids==gold;surface=True
    elif policy=='stepwise':s=len(ids)==1;content=ids==gold[:1];surface=True
    elif policy=='hint':s=bool(text.strip()) and not ids;content=None;surface=text.lstrip().startswith('HINT:')
    else:s=bool(text.strip()) and text.count('?')==1 and not ids;content=None;surface=True
    return dict(strategy_proxy_pass=bool(s),content_exact=content,surface_contract_pass=surface,joint_proxy_pass=bool(s) and content is not False,strict_contract_pass=bool(s) and content is not False and surface,mentioned_ids=ids,helpfulness_evaluated=False)
def raw_input(e,history,consumed,q):
    inp=e.encode(history,q);canonical=inp['input_ids'];prefix=e.serialize(history);assert canonical[:len(prefix)]==prefix
    raw=consumed+canonical[len(prefix):];assert e.tok.decode(raw,skip_special_tokens=False)==e.tok.decode(canonical,skip_special_tokens=False)
    return dict(inp,input_ids=raw,canonical_input_ids=canonical,canonical_assistant_positions=inp['assistant_positions'],assistant_positions=[],boundary_excluded=[],tokenization_mode='retained generated history IDs plus official current-query suffix')
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);cs=json.loads((DATA/'designs.json').read_text());assert len(cs)==6
    cfg=dict(stage='policy_practice_v2',cases=6,history_turns=4,methods=['AOnline','BReplay','CPasted','BSplitNull','BTokenReplay','CStrongRule'],queries=2,history_cap=128,output_cap=256,expected_history_outputs=24,expected_final_outputs=72,total_free_cap=96,
        previous_failed_attempts=3,previous_failure_preserved='results/policy_practice_v1 and logs/policy_practice_v1.log',all_development=True,
        protocol_change='A uses real raw generated history IDs; B uses official text retokenization; BTokenReplay exactly replays A input IDs. Decoded full template strings must match. A/B can differ in history tokenization. BSplitNull uses canonical history.',
        parser_change='Detect each ITEM_[A-H] occurrence even when model concatenates IDs; frozen before v2 outputs.',source='MTI facts with custom tasks, not official benchmark',no_KL=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'designs.json',DATA/'selection.json',ROOT/'results/policy_practice_v1/config.json',ROOT/'results/policy_practice_v1/history_generation_attempts.jsonl']},
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['run_policy_practice_v2.py','build_policy_practice_v1.py','run_policy_threeway_v1.py','migration_exploration_core_v1.py']})
    ch=digest(cfg);write(OUT/'config.json',cfg);write(OUT/'registration.json',dict(config_hash=ch,registered_unix=time.time(),outputs_seen=0));e=OnlineEngine();count=0
    for c in cs:
        history=[];cache=None;consumed=[];scores=[]
        for ti,u in enumerate(c['user_turns']):
            write(STATUS,dict(phase='constructing_history',case_id=c['case_id'],turn=ti,final_outputs=count))
            canonical=e.serialize(history,u);prefix=e.serialize(history);assert canonical[:len(prefix)]==prefix if history else True
            suffix=canonical[len(prefix):] if history else canonical
            actual=consumed+suffix;assert e.tok.decode(actual,skip_special_tokens=False)==e.tok.decode(canonical,skip_special_tokens=False)
            e.calls=[];logits,cache=e.feed(suffix,cache);result,cache,consumed=e.decode(logits,cache,actual.copy(),cfg['history_cap'])
            append(OUT/'history_generation_attempts.jsonl',dict(case_id=c['case_id'],turn=ti,config_hash=ch,**result));assert not result['truncated'],'Incomplete history saved'
            history.extend([dict(role='user',content=u),dict(role='assistant',content=result['text'])])
            tail=e.tok.encode('<|im_end|>\n',add_special_tokens=False);assert result['generated_ids'][-1]==tail[0],'Unexpected terminating token saved'
            _,cache=e.feed(tail,cache);consumed=consumed+tail;complete=e.serialize(history)
            assert e.tok.decode(consumed,skip_special_tokens=False)==e.tok.decode(complete,skip_special_tokens=False) and cache.get_seq_length()==len(consumed)
            s=assess(result['text'],c['policy'],c['history_tasks'][ti]['gold'],False);scores.append(s)
            append(OUT/'history_outputs.jsonl',dict(case_id=c['case_id'],turn=ti,score=s,config_hash=ch,calls=e.calls,canonical_ids=complete,raw_ids=consumed.copy(),retokenization_equal=consumed==complete,**result))
        established=all(s['joint_proxy_pass'] for s in scores);strategy_established=all(s['strategy_proxy_pass'] for s in scores)
        canonical_prefix=e.serialize(history)
        append(OUT/'histories.jsonl',dict(**c,history=history,history_ids=consumed,canonical_history_ids=canonical_prefix,history_scores=scores,history_established_proxy=established,history_strategy_established=strategy_established,config_hash=ch))
        for qname,q in c['queries'].items():
            for m in cfg['methods']:
                write(STATUS,dict(phase='generating',case_id=c['case_id'],query=qname,method=m,final_outputs=count))
                actual=q+'\n\n'+c['strong_rules'][qname] if m=='CStrongRule' else q
                inp=raw_input(e,history,consumed,actual) if m in ['AOnline','BTokenReplay'] else e.encode(history,actual,m in ['CPasted','CStrongRule']);e.calls=[]
                if m=='AOnline':cc=copy.deepcopy(cache);logits,cc=e.feed(inp['input_ids'][len(consumed):],cc)
                elif m=='BSplitNull':_,cc=e.feed(canonical_prefix);logits,cc=e.feed(inp['input_ids'][len(canonical_prefix):],cc)
                else:logits,cc=e.feed(inp['input_ids'])
                first=logits.detach().float().cpu();result,cc,_=e.decode(logits,cc,inp['input_ids'].copy(),cfg['output_cap']);del cc
                append(OUT/'outputs.jsonl',dict(case_id=c['case_id'],query=qname,method=m,input=inp,calls=e.calls,history_established_proxy=established,history_strategy_established=strategy_established,
                    score=assess(result['text'],c['policy'],c['final_task']['gold'],qname=='override_request'),first_argmax=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.numpy().tobytes()).hexdigest(),config_hash=ch,**result));count+=1
        del cache
    assert count==72;write(OUT/'summary.json',dict(phase='completed',history_outputs=24,final_outputs=count,actual_free_outputs=96,previous_failed_free_outputs=3,config_hash=ch));write(STATUS,json.loads((OUT/'summary.json').read_text()))
if __name__=='__main__':
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
