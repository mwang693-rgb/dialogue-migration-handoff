"""Bounded online/replayed/pasted policy continuation pilot, no steering search."""
from migration_exploration_core_v1 import *
import copy, traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/policy_threeway_v1';STATUS=ROOT/'artifacts/policy_threeway_v1_status.json'
SOURCE=ROOT/'data/migration_exploration_v1/cases.jsonl'
CONTRACTS={
 'stepwise':'For this conversation, when I ask for help with remaining work, give only the next pending action, then stop and wait. Use one bullet line per action, beginning with its exact ACT ID. Do not list other actions. A later explicit request for all remaining actions overrides this rule for that reply.',
 'hint':'For this conversation, when I ask for help with remaining work, give only a brief hint and do not reveal any ACT ID. Start the reply with HINT:. A later explicit request to reveal the answer overrides this rule for that reply; then start with ANSWER: and name the next pending ACT ID.',
 'confirm':'For this conversation, when I ask for help with remaining work, ask me one confirmation question before giving an action list. Do not reveal any ACT ID in that question. If I explicitly say to skip confirmation for a reply, list all remaining actions, one bullet line per action beginning with its exact ACT ID.'}
OVERRIDES={
 'stepwise':'For this reply, override our one-step rule and list all remaining approved actions in order, one bullet line per action beginning with its ACT ID.',
 'hint':'For this reply, override our hint-only rule and reveal the answer. Start with ANSWER: and name only the next pending approved ACT ID.',
 'confirm':'For this reply, skip confirmation and list all remaining approved actions in order, one bullet line per action beginning with its ACT ID.'}
def assess(text,c,q):
    ids=re.findall(r'\bACT\d+\b',text);pending=c['pending'];kind=c['policy'];cont=q=='continue'
    lines=[re.match(r'^\s*(?:[-*]|\d+[.)])\s+(ACT\d+)\b',line) for line in text.splitlines()]
    action_lines=[m.group(1) for m in lines if m]
    if kind=='hint':
        strategy=text.lstrip().startswith('HINT:') and not ids if cont else text.lstrip().startswith('ANSWER:')
        content=None if cont else ids==pending[:1]
    elif kind=='confirm' and cont:
        strategy='?' in text and not ids;content=None
    else:
        expected=pending[:1] if kind=='stepwise' and cont else pending
        strategy=len(action_lines)==len(expected) and ids==action_lines;content=action_lines==expected
    return dict(strategy_proxy_pass=bool(strategy),content_exact=content,joint_proxy_pass=bool(strategy) and content is not False,
        mentioned_ids=ids,action_lines=action_lines,withheld_content_not_evaluated=content is None,
        caveat='Question/hint markers and ID withholding/counts are bounded proxies, not full semantic helpfulness or personality.')
class OnlineEngine(Engine):
    def __init__(self):super().__init__();self.calls=[]
    @torch.inference_mode()
    def feed(self,ids,cache=None):
        before=0 if cache is None else cache.get_seq_length();assert ids
        x=torch.tensor([ids],device='cuda');mask=torch.ones((1,before+len(ids)),device='cuda',dtype=torch.long)
        self.calls.append(dict(ids=ids,cache_before=before,attention_length=before+len(ids)))
        out=self.model(input_ids=x,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
        return out.logits[0,-1],out.past_key_values
    @torch.inference_mode()
    def decode(self,logits,cache,consumed,cap):
        generated=[]
        for i in range(cap):
            token=int(logits.argmax());generated.append(token)
            if token in self.eos or i==cap-1:break
            logits,cache=self.feed([token],cache);consumed=consumed+[token]
        natural=generated[-1] in self.eos;raw=generated[:-1] if natural else generated
        return dict(text=self.tok.decode(raw,skip_special_tokens=True),token_ids=raw,generated_ids=generated,truncated=not natural,cap=cap),cache,consumed
    def serialize(self,history,query=None):
        ms=[SYSTEM]+history+([dict(role='user',content=query)] if query is not None else [])
        return self.tok.apply_chat_template(ms,tokenize=True,add_generation_prompt=query is not None)
def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    src=[c for c in rows(SOURCE) if c['family']=='approved'][:6];assert len(src)==6
    cases=[]
    for i,c in enumerate(src):
        policy=['stepwise','hint','confirm'][i%3];truth=c['q1_truth']
        facts='Approved schedule: '+json.dumps(truth['approved_order'])+'. Completed actions: '+json.dumps(truth['completed'])+'.'
        cases.append(dict(case_id='policy-'+c['case_id'],source_case_id=c['case_id'],policy=policy,pending=truth['approved_order'][len(truth['completed']):],
            user_turns=[facts+'\n'+CONTRACTS[policy]+'\nAcknowledge this arrangement without giving an action or revealing an ACT ID.',
                'The schedule and completed actions have not changed. Briefly confirm that you will keep our interaction arrangement; do not give an action or reveal an ACT ID yet.'],
            queries=dict(continue_request='Help me with the remaining approved work.',override_request=OVERRIDES[policy])))
    cfg=dict(stage='policy_threeway_v1',cases=6,policies=3,history_turns=2,methods=['AOnline','BReplay','CPasted','BSplitNull'],queries=2,
        history_cap=128,output_cap=256,expected_history_outputs=12,expected_final_outputs=48,total_free_cap=60,
        source_kind='Existing custom approved-plan tasks, NOT an official public benchmark; new interaction designs using existing facts.',
        selection='first six approved cases, all included regardless of outcomes',all_development=True,
        strategy_metrics='frozen bounded action cardinality, withheld ACT IDs, confirmation question, explicit override; no KL',
        code_sha256=sha(Path(__file__)),core_sha256=sha(ROOT/'scripts/migration_exploration_core_v1.py'),source_sha256=sha(SOURCE))
    ch=digest(cfg);write(OUT/'config.json',cfg);write(OUT/'registration.json',dict(config_hash=ch,registered_unix=time.time(),outputs_seen=0));write(OUT/'designs.json',cases)
    e=OnlineEngine();count=0
    for c in cases:
        history=[];cache=None;consumed=[];turn_checks=[]
        for ti,u in enumerate(c['user_turns']):
            full=e.serialize(history,u);assert full[:len(consumed)]==consumed,'Generated-text token roundtrip mismatch; cannot claim same A/B history'
            e.calls=[];logits,cache=e.feed(full[len(consumed):],cache);consumed=full.copy()
            result,cache,consumed=e.decode(logits,cache,consumed,cfg['history_cap']);assert not result['truncated'],'Incomplete history: preserve failure, do not silently include'
            history.extend([dict(role='user',content=u),dict(role='assistant',content=result['text'])])
            complete=e.serialize(history);assert complete[:len(consumed)]==consumed,'Chat template/generated token identity failed'
            tail=complete[len(consumed):]
            if tail:_,cache=e.feed(tail,cache)
            consumed=complete;assert cache.get_seq_length()==len(consumed)
            turn_checks.append(dict(turn=ti,native_token_identity=True,cache_length=len(consumed)))
            append(OUT/'history_outputs.jsonl',dict(case_id=c['case_id'],turn=ti,config_hash=ch,calls=e.calls,**result))
        append(OUT/'histories.jsonl',dict(**c,history=history,history_ids=consumed,turn_checks=turn_checks,config_hash=ch))
        for qname,q in c['queries'].items():
            native=e.encode(history,q);assert native['input_ids'][:len(consumed)]==consumed
            for method in cfg['methods']:
                write(STATUS,dict(phase='generating',case_id=c['case_id'],query=qname,method=method,outputs=count))
                inp=e.encode(history,q,method=='CPasted');e.calls=[]
                if method=='AOnline':cc=copy.deepcopy(cache);prefix=consumed;logits,cc=e.feed(inp['input_ids'][len(prefix):],cc)
                elif method=='BSplitNull':
                    _,cc=e.feed(consumed);prefix=consumed;logits,cc=e.feed(inp['input_ids'][len(prefix):],cc)
                else:prefix=[];logits,cc=e.feed(inp['input_ids'])
                # Full and split consume identical serialized N IDs; only C changes roles/template.
                first=logits.detach().float().cpu();result,cc,_=e.decode(logits,cc,inp['input_ids'].copy(),cfg['output_cap']);del cc
                append(OUT/'outputs.jsonl',dict(case_id=c['case_id'],query=qname,method=method,input=inp,calls=e.calls,
                    first_argmax=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.numpy().tobytes()).hexdigest(),score=assess(result['text'],c,'continue' if qname=='continue_request' else 'override'),config_hash=ch,**result))
                count+=1
        del cache
    assert count==48;write(OUT/'summary.json',dict(phase='completed',final_outputs=count,history_outputs=12,actual_free_outputs=60,config_hash=ch));write(STATUS,json.loads((OUT/'summary.json').read_text()))
if __name__=='__main__':
    try:main()
    except Exception as ex:write(STATUS,dict(phase='failed',error=repr(ex),traceback=traceback.format_exc()));raise
