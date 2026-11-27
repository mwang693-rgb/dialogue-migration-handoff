"""Freeze the per-layer v2.1 format directions used by the length-direction control (CPU only).

d_L = mean over the 120 v2.1 ordinary pairs of (nat - flat) at the last prompt token, for
layers 4..28. d_16 must equal intervention v1's full_sample_mean, which transfer v1 used.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from confirmation.common import sha, write

LAYERS = [4, 8, 12, 16, 20, 24, 28]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff', type=Path, required=True)
    p.add_argument('--directions', type=Path, required=True, help='intervention v1 directions.pt')
    p.add_argument('--transfer-inputs', type=Path, required=True, help='folder with transfer v1 units.jsonl and vectors.pt')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import torch
    by = defaultdict(dict)
    for r in torch.load(a.handoff / 'activations.pt', map_location='cpu', weights_only=True)['records']:
        by[(int(r['layer']), r['source_id'])][r['condition']] = r['vector'].double()
    d = {L: torch.stack([v['nat'] - v['flat'] for (l, _), v in by.items() if l == L]).mean(0).float() for L in LAYERS}
    ref = torch.load(a.directions, map_location='cpu', weights_only=True)['full_sample_mean'].float()
    assert torch.allclose(d[16], ref, rtol=1e-4, atol=1e-4), float((d[16] - ref).norm())
    a.out.mkdir(parents=True, exist_ok=False)
    torch.save(dict(schema_version=1, layers=LAYERS, d={L: d[L] for L in LAYERS}, source='v2.1 ordinary nat-flat mean'), a.out / 'format_dirs.pt')
    units = [json.loads(s) for s in (a.transfer_inputs / 'units.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    manifest = dict(format_dirs_sha256=sha(a.out / 'format_dirs.pt'), transfer_units_sha256=sha(a.transfer_inputs / 'units.jsonl'),
                    d_norms={L: float(d[L].norm()) for L in LAYERS}, d16_matches_full_sample_mean=True,
                    test_units=sum(u['group'] == 'open' for u in units), direction_units=sum(u['group'] != 'open' for u in units))
    write(a.out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
