"""Shared immutable HF inference, real token scopes, and streaming paired metrics."""
import json, hashlib, time, re, contextlib
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
MODEL='/root/autodl-tmp/models/qwen2.5-7b-instruct'
SYSTEM={'role':'system','content':'You are a helpful assistant.'}
CUE='Continue the conversation.'
HINT='Continue this conversation using its established context. Follow the current request; do not treat superseded requests as still active.'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def append(p,x):
    with Path(p).open('a',encoding='utf-8') as f: f.write(json.dumps(x,ensure_ascii=False)+'\n'); f.flush()
def messages(history,query,pasted=False,hint=False):
    if not pasted: return [SYSTEM]+history+[{'role':'user','content':query}]
    body='\n\n'.join(('User' if m['role']=='user' else 'Assistant')+':\n'+m['content'] for m in history)
    body+='\n\n--- Current user request ---\n'+query
    if hint: body+='\n\n'+HINT
    return [SYSTEM,{'role':'user','content':body}]
def label_parse(text):
    body=text.strip()
    if body.startswith('```') and body.endswith('```'): body=re.sub(r'^```[^\n]*\n|\n```$','',body).strip()
    return body if body in ['A','B'] else None
def score(text,c,q):
    if q=='q2':
        lab=label_parse(text); return dict(label=lab,correct=lab==c['q2_truth'],strict_raw=text.strip()==c['q2_truth'],metric='diagnostic label, not original benchmark')
    if c['family']=='mti':
        body=re.sub(r'^Answer:\s*','',text.strip(),flags=re.I).rstrip('.')
        pred={s.strip().casefold() for s in body.split(',') if s.strip()}; gold={s.casefold() for s in c['q1_truth']['answers']}
        return dict(correct=pred==gold,prediction=sorted(pred),truth=sorted(gold),metric='whole answer set equality')
    from explicit_state_checks_v1 import json_objects
    ids=re.findall(r'\bACT\d+\b',text); truth=c['q1_truth']; claims=[]
    for obj in json_objects(text):
        for key,value in obj.items():
            if key=='approved_order' and isinstance(value,list): claims.append(dict(field=key,conflict=value!=truth['approved_order']))
            elif key in ['completed','completed_actions'] and isinstance(value,list):
                allowed=truth['completed']+[truth['target']]; claims.append(dict(field=key,conflict=any(v not in allowed for v in value)))
            elif key in ['action_id','current_action','latest_action'] and isinstance(value,str): claims.append(dict(field=key,conflict=value!=truth['target']))
    return dict(first_id_proxy_correct=bool(ids) and ids[0]==truth['target'],mentioned_ids=ids,
        explicit_conflict=any(r['conflict'] for r in claims),claims=claims,
        metric='first-ID is descriptive only; partial explicit fields, missing fields inconclusive; next_action not scored after execution')
def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]/max(1,len(a),len(b))
class Engine:
    def __init__(self,load=True):
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); self.model=None
        if load:
            torch.manual_seed(453);torch.set_num_threads(4)
            self.model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
            self.model.requires_grad_(False)
            gc=self.model.generation_config;gc.do_sample=False;gc.repetition_penalty=1.0;gc.temperature=None;gc.top_p=None;gc.top_k=None
            e=gc.eos_token_id;self.eos={e} if isinstance(e,int) else set(e)
    def encode(self,history,query,pasted=False,hint=False):
        ms=messages(history,query,pasted,hint); rendered=self.tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
        encoded=self.tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids'];assert ids==self.tok.apply_chat_template(ms,tokenize=True,add_generation_prompt=True)
        spans=[]; cursor=rendered.index(ms[0]['content'])+len(ms[0]['content'])
        for i,m in enumerate(history):
            start=rendered.find(m['content'],cursor);assert start>=cursor
            end=start+len(m['content']);cursor=end
            spans.append(dict(message_index=i,role=m['role'],char_start=start,char_end=end))
        qs=rendered.find(query,cursor);assert qs>=cursor
        positions=[]; boundary=[]
        for s in spans:
            if s['role']!='assistant':continue
            for i,(a,b) in enumerate(encoded['offset_mapping']):
                if b>a and a>=s['char_start'] and b<=s['char_end']:positions.append(i)
                elif b>a and a<s['char_end'] and b>s['char_start']:boundary.append(i)
        positions=sorted(set(positions)); assert positions and len(ids)<12288
        assert all(encoded['offset_mapping'][i][1]<=qs for i in positions)
        return dict(input_ids=ids,assistant_positions=positions,boundary_excluded=sorted(set(boundary)),
            spans=spans,query_char_start=qs,query_text=query,pasted=pasted,hint=hint,rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest())
    @contextlib.contextmanager
    def hooks(self,inp,intervention=None,capture_layers=()):
        handles=[];captured={};scope=[]
        for index in sorted(set(capture_layers)|({intervention['layer']} if intervention else set())):
            def hook(module,args,out,index=index):
                h=out[0] if isinstance(out,tuple) else out
                if index in capture_layers:captured[index]=h[0,-1].detach().float().cpu()
                if not intervention or index!=intervention['layer']:return out
                kind=intervention['scope']; pos=inp['assistant_positions'] if kind=='HistoryRead' and h.shape[1]==len(inp['input_ids']) else ([h.shape[1]-1] if kind=='Output' else [])
                assert all(0<=p<h.shape[1] for p in pos)
                if not pos or intervention['rho']==0:
                    scope.append(dict(layer=index,sequence_length=h.shape[1],positions=[],rho=intervention['rho']));return out
                z=h.clone();orig=h[:,pos,:];v=intervention['vector'].float();v=v/v.norm()
                inc=orig.float().norm(dim=-1,keepdim=True)*intervention['rho']*v.to(h.device)
                z[:,pos,:]=(orig.float()+inc).to(h.dtype)
                keep=torch.ones(h.shape[1],device=h.device,dtype=torch.bool);keep[pos]=False
                assert torch.equal(z[:,keep,:],h[:,keep,:])
                if kind=='HistoryRead': assert h.shape[1]==len(inp['input_ids']) and max(pos)<len(inp['input_ids'])-1
                actual=(z[:,pos,:].float()-orig.float()).norm(dim=-1)/orig.float().norm(dim=-1)
                scope.append(dict(layer=index,sequence_length=h.shape[1],positions=pos,rho=intervention['rho'],
                    actual_relative_norm_mean=float(actual.mean()),unselected_exact=True))
                return (z,)+out[1:] if isinstance(out,tuple) else z
            handles.append(self.model.model.layers[index].register_forward_hook(hook))
        try:yield captured,scope
        finally:
            for h in handles:h.remove()
    @torch.inference_mode()
    def start(self,inp,intervention=None,capture_layers=()):
        ids=inp['input_ids'];x=torch.tensor([ids],device='cuda')
        with self.hooks(inp,intervention,capture_layers) as (states,scope):
            out=self.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=True,logits_to_keep=1)
        return out.logits[0,-1],out.past_key_values,states,scope
    @torch.inference_mode()
    def next(self,token,cache,inp,intervention=None):
        n=cache.get_seq_length();x=torch.tensor([[token]],device='cuda')
        with self.hooks(inp,intervention) as (_,scope):
            out=self.model(input_ids=x,attention_mask=torch.ones((1,n+1),device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True,logits_to_keep=1)
        return out.logits[0,-1],out.past_key_values,scope
    @torch.inference_mode()
    def generate(self,inp,cap=256,intervention=None):
        torch.cuda.synchronize();t=time.perf_counter(); logits,cache,_,scope=self.start(inp,intervention);gen=[];trace=[]
        for i in range(cap):
            token=int(logits.argmax());gen.append(token)
            trace.append(dict(step=i,cache_length=cache.get_seq_length(),next_token=token))
            if token in self.eos or i==cap-1:break
            logits,cache,new_scope=self.next(token,cache,inp,intervention);scope.extend(new_scope)
        natural=gen[-1] in self.eos;raw=gen[:-1] if natural else gen
        assert all(v['cache_length']==len(inp['input_ids'])+i for i,v in enumerate(trace))
        torch.cuda.synchronize();del cache
        return dict(text=self.tok.decode(raw,skip_special_tokens=True),token_ids=raw,generated_ids=gen,
            truncated=not natural,cap=cap,trace=trace,hook_scope=scope,seconds=time.perf_counter()-t,input=inp)
    @torch.inference_mode()
    def candidate(self,inp,truth,intervention=None):
        logits,cache,_,scope=self.start(inp,intervention);del cache
        probs=torch.log_softmax(logits.double(),dim=-1);values={}
        for label in ['A','B']:
            ids=self.tok.encode(label,add_special_tokens=False); assert len(ids)==1
            values[label]=float(probs[ids[0]])
        alt='B' if truth=='A' else 'A'
        return dict(logprob_sum=values,logprob_average=values,current_minus_previous=values[truth]-values[alt],
            candidate_tokens={s:self.tok.encode(s,add_special_tokens=False) for s in ['A','B']},hook_scope=scope)
    @torch.inference_mode()
    def paired_distribution(self,ninp,pinp,reference,intervention=None,cap=128):
        refs=reference[:cap]; assert refs
        nl,ncache,_,_=self.start(ninp);pl,pcache,_,pscope=self.start(pinp,intervention);records=[]
        for i,token in enumerate(refs):
            nlog=torch.log_softmax(nl.double(),dim=-1);plog=torch.log_softmax(pl.double(),dim=-1)
            nprob=nlog.exp();pprob=plog.exp();mix=torch.logaddexp(nlog,plog)-0.6931471805599453
            kl=(nprob*(nlog-plog)).sum();js=0.5*((nprob*(nlog-mix)).sum()+(pprob*(plog-mix)).sum())
            assert torch.isfinite(kl) and torch.isfinite(js) and float(kl)>=-1e-10 and float(js)>=-1e-10
            records.append(dict(position=i,reference_token=token,kl=float(kl),js=float(js),native_top1=int(nl.argmax()),condition_top1=int(pl.argmax()),top1_same=bool(nl.argmax()==pl.argmax())))
            if i<len(refs)-1:
                nl,ncache,_=self.next(token,ncache,ninp);pl,pcache,s=self.next(token,pcache,pinp,intervention);pscope.extend(s)
        del ncache,pcache
        return dict(records=records,reference_tokens=refs,reference_original_length=len(reference),reference_cropped=len(reference)>cap,
            kl_mean=sum(r['kl'] for r in records)/len(records),js_mean=sum(r['js'] for r in records)/len(records),
            top1_same_fraction=sum(r['top1_same'] for r in records)/len(records),hook_scope=pscope)
