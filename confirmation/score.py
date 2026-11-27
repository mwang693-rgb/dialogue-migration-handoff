"""Check old scoring or summarize the new response-rule batch."""
import argparse
from pathlib import Path
import random
import sys
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT,read,rows,write,check_rule


def bootstrap_ci(values,repeats=2000,seed=453):
    assert values and repeats>0
    rng=random.Random(seed)
    means=sorted(mean(rng.choices(values,k=len(values))) for _ in range(repeats))
    def percentile(q):
        pos=(len(means)-1)*q;i=int(pos);j=min(i+1,len(means)-1)
        return means[i]+(means[j]-means[i])*(pos-i)
    return [percentile(.025),percentile(.975)]


def validate_old():
    from handoff import check_strategy
    directory=ROOT/'experiments/02_strategy'
    histories={h['case_id']:h for h in rows(directory/'histories.jsonl')}
    outputs=rows(directory/'outputs.jsonl')
    for r in outputs: assert check_strategy(r['text'],histories[r['case_id']])==r['source_pass']
    two=[r for r in outputs if r['inst_name']=='combination:two_responses']
    return dict(old_records_verified=len(outputs),old_two_response_records=len(two),source_category_pass=sum(r['source_pass'] for r in two),new_joint_definition_pass=sum(check_rule(r['text'],'ordinary',r['truncated'])['passed'] for r in two),note='Different definitions: old category check versus new exact separator, word limit and complete-stop check. This is a development check, not new confirmation.')


def summarize(folder):
    cfg=read(folder/'protocol.json');outputs=rows(folder/'outputs.jsonl')
    assert outputs and not any(r.get('synthetic',False) for r in outputs), 'Do not report synthetic fixtures as behavior'
    assert all(r.get('run_kind')=='confirmation' for r in outputs), 'Development smoke outputs are not confirmation data'
    from confirmation.validate import validate_run
    validation=validate_run(folder,require_real=True)
    index={(r['source_id'],r['request_type'],r['condition']):r for r in outputs}
    sources=sorted(set(r['source_id'] for r in outputs))
    assert len(index)==len(outputs)
    for r in outputs:
        assert check_rule(r['text'],r['request_type'],r['termination']!='assistant_eos',cfg['word_limit'],r['inst_name'],r['inst_args'])==r['score']
    groups={}
    for q in ['ordinary','change']:
        groups[q]={c:dict(n=len(sources),passed=sum(index[s,q,c]['score']['passed'] for s in sources)) for c in cfg['conditions']}
    differences=[int(index[s,'ordinary','B']['score']['passed'])-int(index[s,'ordinary','C']['score']['passed']) for s in sources]
    delta=mean(differences);ci=bootstrap_ci(differences,cfg['probe']['bootstrap'],cfg['seed'])
    ab_agreement=mean(index[s,q,'A']['score']['passed']==index[s,q,'B']['score']['passed'] for s in sources for q in ['ordinary','change'])
    assert all(index[s,q,'A']['input_ids']==index[s,q,'B']['input_ids'] for s in sources for q in ['ordinary','change'])
    histories=rows(folder/'histories.jsonl');established=mean(h['established'] for h in histories)
    audit=read(folder/'audit_comparison.json') if (folder/'audit_comparison.json').exists() else None
    checks=dict(full_sample=len(sources)==cfg['target_sources'],positive_ci=ci[0]>0)
    strata={}
    for name in cfg['rules']:
        sub=[r for r in outputs if r['inst_name']==name]
        strata[name]={q:{c:dict(n=sum(r['request_type']==q and r['condition']==c for r in sub),passed=sum(r['score']['passed'] for r in sub if r['request_type']==q and r['condition']==c)) for c in cfg['conditions']} for q in ['ordinary','change']}
    return dict(validation=validation,groups=groups,by_rule=strata,ordinary_B_minus_C=delta,source_bootstrap_95ci=ci,AB_behavior_agreement=ab_agreement,established_fraction=established,human_audit=audit,gate_checks=checks,intervention_gate_pass=all(checks.values()),interventions_require_separate_registration=True,probe_may_proceed_if_behavior_gate_fails=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate-old',action='store_true')
    p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.validate_old: print(validate_old())
    else:
        assert a.run
        result=summarize(a.run);write(a.run/'behavior_summary.json',result);print(result)


if __name__=='__main__':main()
