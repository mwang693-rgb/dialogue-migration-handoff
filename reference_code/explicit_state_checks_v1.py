"""Conservative symbolic contradictions; unknown/missing fields inconclusive."""
import json
import re

def json_objects(text):
    candidates=[text] if text.lstrip().startswith(('{','[')) else []
    lines=text.splitlines(keepends=True);i=0
    while i<len(lines):
        m=re.match(r'^\s*(`{3,}|~{3,})([^\n]*)',lines[i]);i+=1
        if not m:continue
        f=m.group(1);body=[]
        while i<len(lines):
            c=re.match(r'^\s*(`{3,}|~{3,})([^\n]*)',lines[i]);line=lines[i];i+=1
            if c and c.group(1)[0]==f[0] and len(c.group(1))>=len(f) and not c.group(2).strip():break
            body.append(line)
        candidates.append(''.join(body))
    objects=[]
    for c in candidates:
        try:o=json.loads(c)
        except (ValueError,TypeError):continue
        if isinstance(o,dict):objects.append(o)
    return objects

def explicit_state(text,h,label):
    order=h['approved_order'] if label=='continue_' else h['comparison_order']
    target=order[2] if label=='continue_' else order[0]
    allowed_completed=order[:3] if label=='continue_' else order[:1]
    claims=[];unknown=[]
    for obj in json_objects(text):
        for k,v in obj.items():
            if k in ['next_action','current_action','action_id','latest_action']:
                if isinstance(v,str) and re.fullmatch(r'ACT\d+',v):claims.append(dict(kind=k,value=v,conflict=v!=target))
                else:unknown.append(k)
            elif k=='approved_order':
                if isinstance(v,list) and all(isinstance(x,str) and re.fullmatch(r'ACT\d+',x) for x in v):claims.append(dict(kind=k,value=v,conflict=v!=order))
                else:unknown.append(k)
            elif k in ['completed','completed_actions']:
                if isinstance(v,list) and all(isinstance(x,str) and re.fullmatch(r'ACT\d+',x) for x in v):
                    claims.append(dict(kind=k,value=v,conflict=any(x not in allowed_completed for x in v)))
                else:unknown.append(k)
    # Only a clear literal action declaration is checked, not arbitrary first ID.
    action=re.match(r'^\s*The next approved action is\s*[`"\']*\s*(ACT\d+)',text)
    if action:claims.append(dict(kind='literal_next_action_declaration',value=action.group(1),conflict=action.group(1)!=target))
    return dict(claims=claims,explicit_claim_present=bool(claims),explicit_state_conflict=any(c['conflict'] for c in claims),unknown_fields=unknown,
        allowed_completed_ids=allowed_completed,target_action=target,
        caveat='No full semantic judge. Missing/unknown fields inconclusive; does not require claiming task completed. Completion may include prior progress plus current action, not later steps.')

def self_test():
    h=dict(approved_order=['ACT0','ACT1','ACT2','ACT3'],comparison_order=['ACT3','ACT2','ACT1','ACT0'])
    assert not explicit_state('```json\n{"completed":["ACT0","ACT1"],"next_action":"ACT2"}\n```',h,'continue_')['explicit_state_conflict']
    assert explicit_state('```json\n{"completed":["ACT2"]}\n```',h,'legal_switch')['explicit_state_conflict']
    assert not explicit_state('Mention ACT2 before choosing ACT3.',h,'legal_switch')['explicit_claim_present']
    assert explicit_state('The next approved action is  `"ACT2"`.',h,'legal_switch')['explicit_state_conflict']
    assert not explicit_state('```json\n{"completed":[]}\n```',h,'legal_switch')['explicit_state_conflict']
    return True
