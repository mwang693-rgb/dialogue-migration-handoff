"""Write the shallow-depth follow-up registration before its generation."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, sha, write

CODE = ['replication/shallow.py', 'replication/freeze_shallow.py', 'replication/engine.py', 'replication/run.py', 'replication/summarize.py',
        'intervention/steer.py', 'transfer/summarize.py', 'confirmation/common.py', 'handoff.py', 'third_party/mt_eval/global_inst.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'replication/protocol_shallow.json')
    p.add_argument('--open-units', type=Path, required=True)
    p.add_argument('--review-note', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists() and read(a.protocol)['status'] == 'reviewed'
    write(a.out, dict(status='registered_before_generation', registered_utc=datetime.now(timezone.utc).isoformat(), review_note=a.review_note,
                      protocol_sha256=sha(a.protocol), open_units_sha256=sha(a.open_units), code_sha256={f: sha(ROOT / f) for f in CODE},
                      outputs_seen='Llama-3.1 k=4 v2.1-style steering units (H1 inconclusive) and Llama baseline/probe results; no k=2 steered output for any model'))


if __name__ == '__main__':
    main()
