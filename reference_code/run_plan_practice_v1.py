"""Two real next-action proposals before format migration; simple known plan state."""
from migration_exploration_core_v1 import *
from run_policy_threeway_v1 import OnlineEngine
from run_policy_practice_v2 import raw_input
import copy,traceback
OUT=ROOT/'results/plan_practice_v1';STATUS=ROOT/'artifacts/plan_practice_v1_status.json';SOURCE=ROOT/'data/migration_exploration_v1/cases.jsonl'
CONTRACT='For this conversation, whenever I ask for help with the approved work, propose only the next pending action and then stop and wait. Do not propose later actions unless I explicitly request all remaining actions for that reply. Name only the action IDs you are proposing now; do not repeat the schedule or completed IDs.'
def designs():
    cs=[c for c in rows(SOURCE) if c['family']=='approved'][:6];assert len(cs)==6;result=[]
    for c in cs:
        order=c['q1_truth']['approved_order'];assert len(order)==4 and len(set(order))==4
        names=json.loads(re.search(r'Actions:\s*(\{.*\})',c['history'][0]['content']).group(1));assert set(names)==set(order)
        u0='Approved schedule: '+json.dumps(order)+'. Action names: '+json.dumps(names)+'. No actions have been completed.\n'+CONTRACT+'\nHelp me start the approved work.'
        u1='I have now completed '+order[0]+'. No other action is completed. Help me continue the approved work.'
        state='I have now completed exactly '+json.dumps(order[:2])+'. No other action is completed. '
        result.append(dict(case_id='plan-practice-'+c['case_id'],source_case_id=c['case_id'],order=order,names=names,user_turns=[u0,u1],history_expected=[[order[0]],[order[1]]],
            queries=dict(continue_request=state+'Help me continue the approved work.',override_request=state+'For this reply, override our one-action rule and list all remaining approved actions in order. Name only the actions you are proposing now.'),
            final_expected=dict(continue_request=order[2:3],override_request=order[2:]),strong_rules=dict(continue_request='For this reply, propose exactly one action: the next pending approved action. Do not name later or completed actions.',override_request='For this reply, propose all remaining actions in approved order, naming no completed actions. The one-action rule does not apply to this reply.')))
    return result
def assess(text,expected):
    ids=re.findall('ACT[0-9]+',text);return dict(proposed_id_proxy=ids,strategy_cardinality_pass=len(ids)==len(expected),content_sequence_pass=ids==expected,joint_pass=ids==expected,
        caveat='ID sequence proxy under an explicit no-restatement contract, not physical execution or full semantic correctness.')
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);cs=designs()
    cfg=dict(stage='plan_practice_v1',cases=6,history_turns=2,methods=['AOnline','BReplay','CPasted','BSplitNull','BTokenReplay','CStrongRule'],queries=2,history_cap=128,output_cap=256,total_free_cap=84,history_outputs=12,final_outputs=72,
        selection='First six existing approved source cases, all included; no final-outcome screening',all_development=True,source='Custom existing approved plan facts, not public benchmark',no_KL=True,
        state_truth='User-declared completion, independent of assistant proposal. History actually generates first/second proposals, not acknowledgements.',
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['run_plan_practice_v1.py','run_policy_threeway_v1.py','run_policy_practice_v2.py','migration_exploration_core_v1.py']},source_sha256=sha(SOURCE))
    ch=digest(cfg);write(OUT/'config.json',cfg);write(OUT/'registration.json',dict(config_hash=ch,outputs_seen=0,registered_unix=time.time()));write(OUT/'designs.json',cs);e=OnlineEngine();count=0
    for c in cs:
        history=[];consumed=[];cache=None;scores=[]
        for ti,u in enumerate(c['user_turns']):
            write(STATUS,dict(phase='constructing_history',case_id=c['case_id'],turn=ti,final_outputs=count))
            canonical=e.serialize(history,u);prefix=e.serialize(history);suffix=canonical[len(prefix):] if history else canonical
            if history:assert canonical[:len(prefix)]==prefix
            actual=consumed+suffix;assert e.tok.decode(actual,skip_special_tokens=False)==e.tok.decode(canonical,skip_special_tokens=False)
            e.calls=[];logits,cache=e.feed(suffix,cache);result,cache,consumed=e.decode(logits,cache,actual,cfg['history_cap']);append(OUT/'history_attempts.jsonl',dict(case_id=c['case_id'],turn=ti,**result));assert not result['truncated']
            history.extend([dict(role='user',content=u),dict(role='assistant',content=result['text'])]);tail=e.tok.encode('<|im_end|>\n',add_special_tokens=False);assert result['generated_ids'][-1]==tail[0]
            _,cache=e.feed(tail,cache);consumed=consumed+tail;canonical=e.serialize(history);assert e.tok.decode(consumed,skip_special_tokens=False)==e.tok.decode(canonical,skip_special_tokens=False)
            s=assess(result['text'],c['history_expected'][ti]);scores.append(s)
            append(OUT/'history_outputs.jsonl',dict(case_id=c['case_id'],turn=ti,raw_ids=consumed.copy(),canonical_ids=canonical,retokenization_equal=consumed==canonical,calls=e.calls,score=s,config_hash=ch,**result))
        established=all(s['joint_pass'] for s in scores);canon_prefix=e.serialize(history)
        append(OUT/'histories.jsonl',dict(**c,history=history,history_ids=consumed,canonical_history_ids=canon_prefix,history_scores=scores,history_established=established,config_hash=ch))
        for qn,q in c['queries'].items():
            for m in cfg['methods']:
                write(STATUS,dict(phase='generating',case_id=c['case_id'],query=qn,method=m,final_outputs=count))
                actual=q+'\n\n'+c['strong_rules'][qn] if m=='CStrongRule' else q
                inp=raw_input(e,history,consumed,actual) if m in ['AOnline','BTokenReplay'] else e.encode(history,actual,m in ['CPasted','CStrongRule']);e.calls=[]
                if m=='AOnline':cc=copy.deepcopy(cache);logits,cc=e.feed(inp['input_ids'][len(consumed):],cc)
                elif m=='BSplitNull':_,cc=e.feed(canon_prefix);logits,cc=e.feed(inp['input_ids'][len(canon_prefix):],cc)
                else:logits,cc=e.feed(inp['input_ids'])
                first=logits.detach().float().cpu();result,cc,_=e.decode(logits,cc,inp['input_ids'].copy(),cfg['output_cap']);del cc
                append(OUT/'outputs.jsonl',dict(case_id=c['case_id'],query=qn,method=m,input=inp,calls=e.calls,score=assess(result['text'],c['final_expected'][qn]),history_established=established,first_argmax=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.numpy().tobytes()).hexdigest(),config_hash=ch,**result));count+=1
        del cache
    assert count==72;write(OUT/'summary.json',dict(phase='completed',actual_free_outputs=84,history_outputs=12,final_outputs=count,config_hash=ch));write(STATUS,json.loads((OUT/'summary.json').read_text()))
if __name__=='__main__':
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
