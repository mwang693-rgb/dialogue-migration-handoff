"""Validate activation shards and their one-to-one link to saved responses."""
from pathlib import Path
from confirmation.common import read, rows


def validate_run(folder, require_real=False):
    import torch
    folder=Path(folder)
    cfg=read(folder/'protocol.json')
    outputs=rows(folder/'outputs.jsonl')
    index={}
    for r in outputs:
        key=(r['case_id'],r['query_id'],r['condition'])
        assert key not in index, 'Duplicate output key'
        index[key]=r
    ids=set(r['case_id'] for r in outputs)
    assert len(outputs)==len(ids)*8, 'Expect two requests and four conditions per valid history'
    for case in ids:
        assert {(r['request_type'],r['condition']) for r in outputs if r['case_id']==case}=={(q,c) for q in ['ordinary','change'] for c in cfg['conditions']}
    n=0; seen=set();dimensions=set();synthetic=[];record_index={};kinds=set()
    for f in sorted((folder/'activations').glob('*.pt')):
        payload=torch.load(f,map_location='cpu',weights_only=True)
        assert payload['schema_version']==1
        kinds.add(payload.get('run_kind','synthetic' if payload['is_synthetic'] else 'unknown'))
        synthetic.append(payload['is_synthetic'])
        if require_real: assert not payload['is_synthetic'], 'Synthetic interface data is not experimental evidence'
        for r in payload['records']:
            key=(r['case_id'],r['query_id'],r['condition'],r['layer'])
            assert key not in seen;seen.add(key)
            assert r['condition'] in ['nat','flat']
            method='B' if r['condition']=='nat' else 'C'
            assert r.get('run_condition',method)==method
            output=index[r['case_id'],r['query_id'],method]
            assert r['source_id']==output['source_id'] and r['request_type']==output['request_type']
            assert r['n_tokens']==len(output['input_ids']) and r['position']==r['n_tokens']-1
            assert r['terminal_token_id']==output['input_ids'][-1]
            v=r['vector']
            assert v.device.type=='cpu' and v.dtype==torch.float32 and v.shape==(payload['hidden_size'],) and torch.isfinite(v).all()
            assert r['layer'] in cfg['layers'];dimensions.add(v.numel());n+=1
            record_index[key]=r
    expected={(c,q,k,l) for c in ids for q in ['ordinary','change'] for k in ['nat','flat'] for l in cfg['layers']}
    assert seen==expected and len(dimensions)==1, 'Missing layer/condition/query or mixed dimensions'
    for case in ids:
        for q in ['ordinary','change']:
            assert index[case,q,'A']['input_ids']==index[case,q,'B']['input_ids']
            nat,flat=record_index[case,q,'nat',0],record_index[case,q,'flat',0]
            assert nat['terminal_token_id']==flat['terminal_token_id']
            assert torch.equal(nat['vector'],flat['vector']), 'Token-embedding control differs'
    assert len(kinds)==1, 'Mixed run provenance'
    return dict(valid=True,synthetic=all(synthetic),run_kind=next(iter(kinds)),cases=len(ids),outputs=len(outputs),activation_records=n,layers=cfg['layers'],hidden_size=next(iter(dimensions)))
