"""All remaining public questions on BOTH existing two-response histories."""
from run_mteval_continuity_v1 import checkers,score as source_score,CLASSES,DATA
from migration_exploration_core_v1 import *
from run_policy_threeway_v1 import OnlineEngine
from run_existing_continuity_v3 import termination
import traceback
BASE=ROOT/'results/mteval_continuity_v1';OUT=ROOT/'results/mteval_tworesponse_extension_v2';STATUS=ROOT/'artifacts/mteval_tworesponse_extension_v2_status.json'
def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    hs=[h for h in rows(BASE/'histories.jsonl') if h['inst_name']=='combination:two_responses'];assert len(hs)==2
    raw=[json.loads(x) for x in (DATA/'mteval_global_multiturn.jsonl').read_text().splitlines()];src={x['id']:x for x in raw}
    tasks=[dict(case_id=h['case_id'],source_id=h['source_id'],query_index=i,query=src[h['source_id']]['conv'][i]['user']) for h in hs for i in range(6,11)];assert len(tasks)==10
    cfg=dict(stage='mteval_tworesponse_extension_v2',methods=['BReplay','CPasted','BSplitNull','CSpecificRule'],max_new_free_outputs=40,cap=1536,all_development=True,
        selection='Both previously frozen two-response histories, ALL remaining source questions6..10, fixed after initial gap; no new history and no online-cache claim. Related queries not independent samples.',
        previous_failed_free_attempts=1,previous_failure='v1 first completed decode not saved because wildcard import shadowed scorer; original config/code/log/status preserved; raw output unavailable',purpose='Does two-version response loss extend across current tasks after three actual practice replies? No steering or parameter search.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'histories.jsonl',BASE/'analysis.json',BASE/'outputs.jsonl',DATA/'mteval_global_multiturn.jsonl']},
        code_sha256={f:sha(ROOT/'scripts'/f) for f in ['run_mteval_tworesponse_extension_v2.py','run_mteval_continuity_v1.py','run_policy_threeway_v1.py','migration_exploration_core_v1.py']})
    ch=digest(cfg);write(OUT/'config.json',cfg);write(OUT/'tasks.json',tasks);write(OUT/'registration.json',dict(config_hash=ch,outputs_seen=0,registered_unix=time.time()));e=OnlineEngine();e.eos.add(e.tok.convert_tokens_to_ids('<|im_start|>'));hc={h['case_id']:h for h in hs};count=0
    for t in tasks:
        h=hc[t['case_id']];prefix=e.serialize(h['history'])
        for m in cfg['methods']:
            write(STATUS,dict(phase='generating',case_id=h['case_id'],query=t['query_index'],method=m,new_outputs=count))
            q=t['query']+'\n\nFor this reply, continue following this conversation-wide response rule: '+h['rule_text'] if m=='CSpecificRule' else t['query']
            inp=e.encode(h['history'],q,m in ['CPasted','CSpecificRule']);e.calls=[]
            if m=='BSplitNull':_,cache=e.feed(prefix);logits,cache=e.feed(inp['input_ids'][len(prefix):],cache)
            else:logits,cache=e.feed(inp['input_ids'])
            first=int(logits.argmax());r,cache,_=e.decode(logits,cache,inp['input_ids'].copy(),1536);del cache;count+=1
            append(OUT/'outputs.jsonl',dict(case_id=t['case_id'],query_index=t['query_index'],method=m,input=inp,calls=e.calls,source_pass=source_score(r['text'],h),word_count=len(re.findall(r'\w+',r['text'])),first_argmax=first,termination=termination(e,r),config_hash=ch,**r))
    assert count==40;summary=dict(phase='completed',new_free_outputs=count,existing_histories=2,queries_per_history=5,config_hash=ch);write(OUT/'summary.json',summary);write(STATUS,summary)
if __name__=='__main__':
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
