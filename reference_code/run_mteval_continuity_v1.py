"""Frozen public MT-Eval response organization, actual three-turn practice."""
from migration_exploration_core_v1 import *
from run_policy_threeway_v1 import OnlineEngine
from run_policy_practice_v2 import raw_input
from run_existing_continuity_v3 import termination
import ast,copy,traceback
DATA=ROOT/'data/continuity_dataset_review_2026_10_14'
SOURCE=DATA/'mteval_continuity_selection_v1.json'
OUT=ROOT/'results/mteval_continuity_v1'
STATUS=ROOT/'artifacts/mteval_continuity_v1_status.json'
CLASSES={'format:json_format':'JsonFormat','format:bullet_list':'BulletList','combination:two_responses':'TwoResponses','startend:end_phrase':'EndPhrase'}

def checkers():
    tree=ast.parse((DATA/'KwanWaiChung__MT-Eval__utils__global_inst.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['Instruction']+list(CLASSES.values())]
    env=dict(json=json,re=re,List=list,Dict=dict,Any=object)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_mteval_checker','exec'),env)
    return env

def score(text,c):
    return bool(checkers()[CLASSES[c['inst_name']]](**c['inst_args']).check_following(text))

def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    cs=json.loads(SOURCE.read_text())['cases'];assert len(cs)==8
    cfg=dict(stage='mteval_continuity_v1',cases=8,history_turns=4,actual_practice_turns=3,queries=2,methods=['AOnline','BReplay','CPasted','BSplitNull','CSpecificRule'],history_cap=1536,output_cap=1536,max_new_free_outputs=112,all_development=True,
        source_adaptation='Original initial user and questions 1..3 form generated history, questions 4 and 5 tested independently. Initial do_inference=false acknowledgement regenerated; original reference assistants never fed. Fixed four categories, both source versions, no outcome screening.',
        history_established='All THREE actual task replies pass source category checker and natural stop; acknowledgement reported separately.',
        score_limit='Source category checker only; not full response utility/semantics, nor secondary under-200-word requirement. TwoResponses checks separated text, not semantic distinctness. Ending phrase is weak surface control.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,DATA/'mteval_global_multiturn.jsonl',DATA/'KwanWaiChung__MT-Eval__utils__global_inst.py']},
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['run_mteval_continuity_v1.py','run_policy_threeway_v1.py','run_policy_practice_v2.py','run_existing_continuity_v3.py','migration_exploration_core_v1.py']})
    ch=digest(cfg);write(OUT/'config.json',cfg);write(OUT/'registration.json',dict(config_hash=ch,outputs_seen=0,registered_unix=time.time()))
    e=OnlineEngine();e.eos.add(e.tok.convert_tokens_to_ids('<|im_start|>'));hc=fc=0
    for c in cs:
        history=[];consumed=[];cache=None;histpass=[];invalid=None
        for ti,u in enumerate(c['history_users']):
            write(STATUS,dict(phase='history',case_id=c['case_id'],turn=ti,history_outputs=hc,final_outputs=fc))
            full=e.serialize(history,u);canonical=e.serialize(history);suffix=full[len(canonical):] if history else full
            assert full[:len(canonical)]==canonical if history else True
            actual=consumed+suffix
            assert e.tok.decode(actual,skip_special_tokens=False)==e.tok.decode(full,skip_special_tokens=False)
            e.calls=[];logits,cache=e.feed(suffix,cache);r,cache,consumed=e.decode(logits,cache,actual,1536);hc+=1
            stop=termination(e,r);special=sorted(set(r['token_ids']).intersection(e.tok.all_special_ids))
            append(OUT/'history_attempts.jsonl',dict(case_id=c['case_id'],turn=ti,calls=e.calls.copy(),termination=stop,special_body_ids=special,config_hash=ch,**r))
            if stop!='assistant_eos' or special:invalid=stop if not special else 'special_body_token';break
            history.extend([dict(role='user',content=u),dict(role='assistant',content=r['text'])])
            tail=e.tok.encode('<|im_end|>\n',add_special_tokens=False);_,cache=e.feed(tail,cache);consumed+=tail
            assert e.tok.decode(consumed,skip_special_tokens=False)==e.tok.decode(e.serialize(history),skip_special_tokens=False)
            passed=score(r['text'],c);histpass.append(passed)
            append(OUT/'history_outputs.jsonl',dict(case_id=c['case_id'],turn=ti,source_pass=passed,raw_ids=consumed.copy(),canonical_ids=e.serialize(history),calls=e.calls.copy(),config_hash=ch,**r))
        if invalid:
            append(OUT/'case_status.jsonl',dict(case_id=c['case_id'],status='invalid_history',turn=ti,reason=invalid));del cache;continue
        established=all(histpass[1:]);h=dict(**c,history=history,history_ids=consumed,canonical_history_ids=e.serialize(history),history_pass=histpass,established=established,config_hash=ch);append(OUT/'histories.jsonl',h)
        for qi,q in enumerate(c['queries']):
            for m in cfg['methods']:
                write(STATUS,dict(phase='final',case_id=c['case_id'],query=qi,method=m,history_outputs=hc,final_outputs=fc))
                actualq=q+'\n\nFor this reply, continue following this conversation-wide response rule: '+c['rule_text'] if m=='CSpecificRule' else q
                inp=raw_input(e,history,consumed,actualq) if m=='AOnline' else e.encode(history,actualq,m in ['CPasted','CSpecificRule']);e.calls=[]
                if m=='AOnline':cc=copy.deepcopy(cache);logits,cc=e.feed(inp['input_ids'][len(consumed):],cc)
                elif m=='BSplitNull':_,cc=e.feed(e.serialize(history));logits,cc=e.feed(inp['input_ids'][len(e.serialize(history)):],cc)
                else:logits,cc=e.feed(inp['input_ids'])
                first=int(logits.argmax());r,cc,_=e.decode(logits,cc,inp['input_ids'].copy(),1536);del cc;fc+=1
                append(OUT/'outputs.jsonl',dict(case_id=c['case_id'],inst_name=c['inst_name'],query=qi,method=m,input=inp,calls=e.calls,source_pass=score(r['text'],c),termination=termination(e,r),first_argmax=first,config_hash=ch,**r))
        del cache;append(OUT/'case_status.jsonl',dict(case_id=c['case_id'],status='completed',established=established))
    assert hc+fc<=112;summary=dict(phase='completed',history_outputs=hc,final_outputs=fc,new_free_outputs=hc+fc,valid_histories=len(rows(OUT/'histories.jsonl')),intended_cases=8,config_hash=ch)
    write(OUT/'summary.json',summary);write(STATUS,summary)
if __name__=='__main__':
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
