"""Generate interface fixtures, explicitly marked synthetic; no model loading."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT,read,write,append,check_rule


def make_fake(out,sources=16,hidden_size=3584):
    import torch
    out=Path(out)
    assert not out.exists()
    (out/'activations').mkdir(parents=True)
    cfg=read(ROOT/'confirmation/protocol.json');write(out/'protocol.json',cfg)
    rng=torch.Generator().manual_seed(cfg['seed'])
    for i in range(sources):
        cid=f'fake-{i:03d}';vectors=[]
        for q in ['ordinary','change']:
            for condition in cfg['conditions']:
                ids=[10+i]* (40+i+(12 if condition in ['C','C_reminder'] else 0))+[198]
                text='First mock response. ****** Second mock response.' if q=='ordinary' else 'One mock response.'
                append(out/'outputs.jsonl',dict(synthetic=True,source_id=cid,case_id=cid,query_id=q,request_type=q,condition=condition,query_text='Synthetic interface fixture only.',input_ids=ids,text=text,truncated=False,termination='assistant_eos',score=check_rule(text,q)))
                if condition in ['B','C']:
                    for layer in cfg['layers']:
                        value=torch.zeros(hidden_size) if layer==0 else torch.randn(hidden_size,generator=rng)
                        vectors.append(dict(source_id=cid,case_id=cid,query_id=q,request_type=q,condition='nat' if condition=='B' else 'flat',run_condition=condition,layer=layer,vector=value,n_tokens=len(ids),position=len(ids)-1,terminal_token_id=ids[-1]))
        torch.save(dict(schema_version=1,is_synthetic=True,hidden_size=hidden_size,records=vectors),out/'activations'/f'{cid}.pt')
    from confirmation.validate import validate_run
    result=validate_run(out);write(out/'interface_check.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sources',type=int,default=16)
    p.add_argument('--hidden-size',type=int,default=3584)
    a=p.parse_args();assert a.sources>0 and a.hidden_size>0
    print(make_fake(a.out,a.sources,a.hidden_size))


if __name__=='__main__':main()
