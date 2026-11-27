"""Write the transfer registration before any new inference."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, sha, write

CODE = ['transfer/build_units.py', 'transfer/run.py', 'transfer/summarize.py', 'transfer/freeze.py',
        'intervention/steer.py', 'confirmation/common.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'transfer/protocol.json')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--review-note', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists(), 'Use a new registration path'
    cfg, manifest = read(a.protocol), read(a.inputs / 'manifest.json')
    assert cfg['status'] == 'reviewed' and manifest['n_units'] == cfg['budget']['units']
    assert manifest['units_sha256'] == sha(a.inputs / 'units.jsonl') and manifest['vectors_sha256'] == sha(a.inputs / 'vectors.pt')
    write(a.out, dict(status='registered_before_generation', registered_utc=datetime.now(timezone.utc).isoformat(),
                      review_note=a.review_note, protocol_sha256=sha(a.protocol), units_sha256=manifest['units_sha256'],
                      vectors_sha256=manifest['vectors_sha256'], manifest=manifest,
                      code_sha256={f: sha(ROOT / f) for f in CODE}, outputs_seen=0))


if __name__ == '__main__':
    main()
