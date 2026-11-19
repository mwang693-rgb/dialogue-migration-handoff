"""Merge validated activation shards and copy the response file for handoff."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmation.common import write,sha
from confirmation.validate import validate_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-synthetic',action='store_true')
    a=p.parse_args();assert not a.out.exists()
    checked=validate_run(a.run,require_real=not a.allow_synthetic)
    import torch
    records=[]
    for file in sorted((a.run/'activations').glob('*.pt')):
        records.extend(torch.load(file,map_location='cpu',weights_only=True)['records'])
    a.out.mkdir(parents=True)
    for kind,name in [('ordinary','activations.pt'),('change','activations_change.pt')]:
        selected=[r for r in records if r['request_type']==kind]
        torch.save(dict(schema_version=1,is_synthetic=checked['synthetic'],run_kind=checked['run_kind'],request_type=kind,hidden_size=checked['hidden_size'],records=selected),a.out/name)
    for name in ['outputs.jsonl','cases.jsonl','histories.jsonl','protocol.json','registration.json','case_status.jsonl']:
        if (a.run/name).exists(): shutil.copyfile(a.run/name,a.out/name)
    write(a.out/'validation.json',checked)
    write(a.out/'files.json',{f.name:sha(f) for f in a.out.iterdir() if f.is_file()})
    print(checked)


if __name__=='__main__':main()
