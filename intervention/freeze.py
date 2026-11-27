"""Write the intervention registration before any new inference."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, sha, write

CODE = ['intervention/run.py', 'intervention/steer.py', 'intervention/directions.py', 'intervention/summarize.py',
        'intervention/freeze.py', 'confirmation/common.py', 'handoff.py']
V21 = ['histories.jsonl', 'outputs.jsonl', 'protocol.json']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'intervention/protocol.json')
    p.add_argument('--directions', type=Path, required=True)
    p.add_argument('--v21', type=Path, required=True)
    p.add_argument('--review-note', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists(), 'Use a new registration path'
    assert read(a.protocol)['status'] == 'reviewed'
    summary = read(a.directions.with_suffix('.json'))
    assert summary['directions_sha256'] == sha(a.directions)
    write(a.out, dict(status='registered_before_generation', registered_utc=datetime.now(timezone.utc).isoformat(),
                      review_note=a.review_note, protocol_sha256=sha(a.protocol), directions_sha256=sha(a.directions),
                      directions_summary=summary, v21_sha256={f: sha(a.v21 / f) for f in V21},
                      code_sha256={f: sha(ROOT / f) for f in CODE}, outputs_seen=0))


if __name__ == '__main__':
    main()
