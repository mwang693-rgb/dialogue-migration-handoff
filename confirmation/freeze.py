"""Create a local registration once the team has reviewed protocol and new cases."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, rows, sha, validate_cases, write, normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT/'confirmation/protocol.json')
    p.add_argument('--review-note', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists(), 'Use a new registration path'
    cfg, cases = read(a.protocol), rows(a.cases)
    assert cfg['status']=='reviewed', 'Team review must change protocol status from draft to reviewed before registration'
    validate_cases(cases, cfg, confirmation=True)
    old = set()
    for f in (ROOT/'experiments').glob('*/**/*histories.jsonl'):
        for h in rows(f):
            old.update(normalize(m['content']) for m in h.get('history',[]) if m['role']=='user')
            queries = h.get('queries', [])
            if isinstance(queries,dict): queries=list(queries.values())
            for q in queries:
                old.add(normalize(q if isinstance(q,str) else q.get('text','')))
    for c in cases:
        for q in c['history_questions']+[c['question']]:
            assert not any(normalize(q) in t for t in old if t), 'Question occurs in exported development histories'
    files = sorted((ROOT/'confirmation').glob('*.py'))+sorted((ROOT/'analysis').glob('*.py'))+[ROOT/'PREREGISTRATION.md',ROOT/'handoff.py',ROOT/'reference_code/migration_exploration_core_v1.py', ROOT/'reference_code/run_policy_threeway_v1.py']
    write(a.out,dict(status='registered_before_generation',registered_utc=datetime.now(timezone.utc).isoformat(),review_note=a.review_note,
                    protocol_sha256=sha(a.protocol), cases_sha256=sha(a.cases), sources=len(cases),
                    code_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
                    novelty_check='Question substring check against exported development histories only; provenance still requires team review.'))


if __name__ == '__main__': main()
