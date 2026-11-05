"""Frozen literal metric, format manipulations, and whole-batch feasibility gate."""
from style_cue_verification_common_v1 import CONTRACTIONS,plaintext
from pathlib import Path
import copy,hashlib,json,math,re,statistics
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/style_confirmation_v2'
CONF=ROOT/'results/style_confirmation_v2'
STEER=ROOT/'results/style_confirmation_steering_v2'
CUE='Continue the conversation.'
LAYER=16
RHO=0.05
STEER_METHODS=['DHistory','DOutput','RandomOutput','ReverseOutput','RestateHint','SpellOutHint']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def features(text):
    n=sum(len(re.findall(r'\b'+re.escape(short).replace("'","['’]")+r'\b',text,re.I)) for _,short in CONTRACTIONS)
    words=len(re.findall(r'\w+',text));return dict(count=n,words=words,per100=100*n/max(1,words))
def assess(text):return dict(full=features(text),first120=features(plaintext(text)))
def variant(history,clause,remove=False):
    result=copy.deepcopy(history);assert result[0]['role']=='user' and result[0]['content'].startswith(clause)
    if remove:result[0]['content']=result[0]['content'][len(clause):]
    assert result[1:]==history[1:];return result

def analyze_confirmation():
    hh=rows(CONF/'histories.jsonl') if (CONF/'histories.jsonl').exists() else []
    rr=rows(CONF/'outputs.jsonl') if (CONF/'outputs.jsonl').exists() else []
    hr=rows(CONF/'history_outputs.jsonl') if (CONF/'history_outputs.jsonl').exists() else []
    index={(r['case_id'],r['query'],r['method']):r for r in rr};groups={};normal=[r for r in rr if r['query'] in ['q0','q1']]
    for view in ['full','first120']:
        per=[]
        for h in hh:
            cid=h['case_id'];mean={m:statistics.mean(index[cid,q,m]['score'][view]['per100'] for q in ['q0','q1']) for m in ['AOnline','BKeep','CKeep','BRemove','CRemove']}
            per.append(dict(case_id=cid,source_id=h['source_id'],mean_rates=mean,c_minus_a=mean['CKeep']-mean['AOnline'],c_minus_b=mean['CKeep']-mean['BKeep'],remove_c_minus_b=mean['CRemove']-mean['BRemove'],interaction=(mean['CKeep']-mean['BKeep'])-(mean['CRemove']-mean['BRemove'])))
        groups[view]=dict(histories=len(per),source_rows=per,
                          means={k:statistics.mean(p[k] for p in per) if per else None for k in ['c_minus_a','c_minus_b','remove_c_minus_b','interaction']},
                          positive_C_minus_A=sum(p['c_minus_a']>0 for p in per),positive_C_minus_B=sum(p['c_minus_b']>0 for p in per),
                          total_counts={m:sum(r['score'][view]['count'] for r in normal if r['method']==m) for m in ['AOnline','BKeep','CKeep','BRemove','CRemove']})
    histories_with_feature=sum(any(r['case_id']==h['case_id'] and r['score']['full']['count']>0 for r in hr) for h in hh)
    reasons=[];n=len(hh);need=math.ceil(.75*n);f=groups['full'];head=groups['first120']
    if n!=4:reasons.append('Not all four frozen histories valid; do not replace cases')
    if histories_with_feature<3:reasons.append('Fewer than three histories actually exhibit the listed feature')
    if any(r['termination']!='assistant_eos' for r in normal):reasons.append('Normal final answer has cap or non-assistant stop')
    for label,g in [('full',f),('first120',head)]:
        for effect in ['c_minus_a','c_minus_b']:
            if g['means'][effect] is None or g['means'][effect]<=0:reasons.append(label+' mean '+effect+' not positive')
    for key in ['positive_C_minus_A','positive_C_minus_B']:
        if f[key]<need:reasons.append('full '+key+' below three-quarter source threshold')
    if head['positive_C_minus_B']<need:reasons.append('first120 positive_C_minus_B below three-quarter threshold')
    if f['total_counts']['CKeep']-f['total_counts']['BKeep']<4:reasons.append('Fewer than four additional actual listed short forms in C vs B')
    if f['means']['interaction'] is None or f['means']['interaction']<=0:reasons.append('Cue interaction not positive')
    overrides=[r for r in rr if r['query']=='override_q1']
    return dict(groups=groups,history_feature_present=histories_with_feature,valid_histories=n,final_outputs=len(rr),
                natural=sum(r['termination']=='assistant_eos' for r in rr),cap=sum(r['termination']=='length_cap' for r in rr),
                override_zero_short_forms={m:sum(r['score']['full']['count']==0 for r in overrides if r['method']==m) for m in ['AOnline','BKeep','CKeep']},
                gate_pass=not reasons,gate_reasons=reasons,gate_is_heuristic_not_significance=True,
                source_groups_not_queries_are_units=True)
