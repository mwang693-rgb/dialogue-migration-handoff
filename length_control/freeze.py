"""Write the length-control registration before any new inference."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT, read, sha, write

CODE = ['length_control/build.py', 'length_control/run.py', 'length_control/summarize.py', 'length_control/freeze.py',
        'transfer/summarize.py', 'transfer/protocol.json', 'intervention/steer.py', 'confirmation/common.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'length_control/protocol.json')
    p.add_argument('--inputs', type=Path, required=True, help='folder with format_dirs.pt and manifest.json')
    p.add_argument('--transfer-inputs', type=Path, required=True)
    p.add_argument('--review-note', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    m = read(a.inputs / 'manifest.json')
    assert m['format_dirs_sha256'] == sha(a.inputs / 'format_dirs.pt') and m['transfer_units_sha256'] == sha(a.transfer_inputs / 'units.jsonl')
    assert read(a.protocol)['status'] == 'reviewed'
    write(a.out, dict(status='registered_before_generation', registered_utc=datetime.now(timezone.utc).isoformat(), review_note=a.review_note,
                      protocol_sha256=sha(a.protocol), units_sha256=m['transfer_units_sha256'], format_dirs_sha256=m['format_dirs_sha256'],
                      manifest=m, code_sha256={f: sha(ROOT / f) for f in CODE}, outputs_seen=0))


if __name__ == '__main__':
    main()
