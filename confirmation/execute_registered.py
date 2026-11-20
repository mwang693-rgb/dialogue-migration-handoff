"""Run a frozen batch once, then validate, score and pack; never retry generation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT,read,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--snapshot-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--registration',type=Path,default=ROOT/'data/confirmation_v2/registration.json')
    a=p.parse_args()
    out=a.out.resolve();assert not out.exists()
    packed=out.with_name(out.name+'-handoff');assert not packed.exists()
    state=out.with_name(out.name+'-supervisor.json')
    log=out.with_name(out.name+'-console.log')
    assert not state.exists() and not log.exists()
    out.parent.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(state,dict(phase='starting',git_commit=commit,started_utc=datetime.now(timezone.utc).isoformat()))
    commands=[
        [sys.executable,str(ROOT/'confirmation/run_batch.py'),'--cases',str(ROOT/'data/confirmation_v2/candidates.jsonl'),'--registration',str(a.registration),'--model',str(a.model),'--snapshot-manifest',str(a.snapshot_manifest),'--out',str(out)],
        [sys.executable,str(ROOT/'confirmation/score.py'),'--run',str(out)],
        [sys.executable,str(ROOT/'confirmation/pack.py'),'--run',str(out),'--out',str(packed)],
    ]
    try:
        with log.open('x',encoding='utf-8') as stream:
            for phase,cmd in zip(['generation','validation_and_scoring','packing'],commands):
                write(state,dict(phase=phase,git_commit=commit,run_directory=str(out)))
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        archive=out.with_suffix('.tar.gz');assert not archive.exists()
        # Includes failed attempts, actual tokens, raw tensors and immutable code.
        with tarfile.open(archive,'w:gz') as tar:
            tar.add(out,arcname=out.name)
            tar.add(packed,arcname=packed.name)
            reg=read(a.registration)
            for rel,digest in reg['code_sha256'].items():
                assert sha(ROOT/rel)==digest
                tar.add(ROOT/rel,arcname='code/'+rel)
            for rel in ['data/confirmation_v2/source.json','data/confirmation_v2/LICENSE.md','THIRD_PARTY_NOTICES.md']:
                tar.add(ROOT/rel,arcname='code/'+rel)
        write(archive.with_name(archive.name+'.manifest.json'),dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,git_commit=commit,local_backup_verified=False))
        write(state,dict(phase='completed_remote_backup',archive=str(archive),sha256=sha(archive),git_commit=commit,local_backup_verified=False))
    except Exception as exc:
        write(state,dict(phase='failed',error=repr(exc),git_commit=commit,run_directory=str(out),automatic_retry=False))
        raise


if __name__=='__main__':main()
