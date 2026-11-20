"""Prepare two already-seen MT-Eval cases for a CUDA smoke test, not confirmation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT,rows,append,validate_cases,read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert not a.out.exists()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    old=[h for h in rows(ROOT/'experiments/02_strategy/histories.jsonl') if h['inst_name']=='combination:two_responses']
    cases=[dict(source_id='dev-'+h['case_id'],case_id='dev-'+h['case_id'],inst_name=h['inst_name'],inst_args=h['inst_args'],rule_text=h['rule_text'],change_text='For this reply only, give one response. Do not use the ****** separator or give a second version.',history_questions=h['history_users'][1:4],question=h['queries'][0],provenance='Already-seen MT-Eval source '+h['source_id']+'; development fixture only',development_fixture=True) for h in old]
    assert len(cases)==2
    validate_cases(cases,read(ROOT/'confirmation/protocol.json'))
    for c in cases:append(a.out,c)
    print({'development_cases':len(cases),'confirmation_cases':0})


if __name__=='__main__':main()
