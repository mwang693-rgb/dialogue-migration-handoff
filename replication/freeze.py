"""Write the replication registration before any new-model generation."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, sha, write

CODE = ['replication/engine.py', 'replication/run.py', 'replication/summarize.py', 'replication/build_open.py', 'replication/freeze.py',
        'intervention/steer.py', 'transfer/summarize.py', 'confirmation/common.py', 'handoff.py', 'third_party/mt_eval/global_inst.py']
V21 = ['cases.jsonl', 'protocol.json']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'replication/protocol.json')
    p.add_argument('--inputs', type=Path, required=True, help='folder with open_units.jsonl and manifest.json')
    p.add_argument('--v21', type=Path, required=True)
    p.add_argument('--review-note', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    m = read(a.inputs / 'manifest.json')
    assert m['open_units_sha256'] == sha(a.inputs / 'open_units.jsonl') and read(a.protocol)['status'] == 'reviewed'
    write(a.out, dict(status='registered_before_generation', registered_utc=datetime.now(timezone.utc).isoformat(), review_note=a.review_note,
                      protocol_sha256=sha(a.protocol), open_units_sha256=m['open_units_sha256'], open_manifest=m,
                      v21_sha256={f: sha(a.v21 / f) for f in V21}, code_sha256={f: sha(ROOT / f) for f in CODE}, outputs_seen=0))


if __name__ == '__main__':
    main()
