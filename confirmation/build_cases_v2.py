"""Select new question bundles before model generation; never inspect responses."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confirmation.common import ROOT,rows,write,append,normalize

REVISION='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a'
EXPECTED='2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec'
RULES=[
 ('combination:two_responses',{},'All your responses in this conversation must contain two different responses separated by exactly six asterisks: ******.','For this reply only, give one response. Do not use the ****** separator or give a second version.'),
 ('format:json_format',{},'All your responses in this conversation must be valid JSON.','For this reply only, answer in plain prose, not JSON. Do not use a code block.'),
 ('format:bullet_list',{'num_bullets':0},'All your responses in this conversation must use a bullet list.','For this reply only, answer in one plain paragraph. Do not use a bulleted or numbered list.'),
 ('startend:end_phrase',{'end_phrase':'Is there anything else I can help with?'},'All your responses in this conversation must end with this exact phrase: Is there anything else I can help with? No text may follow it.','For this reply only, omit the usual ending phrase "Is there anything else I can help with?". End with the answer itself.')
]


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists();a.out.mkdir(parents=True)
 cache=ROOT/'runs/source-cache/dolly.jsonl';cache.parent.mkdir(parents=True,exist_ok=True)
 url=f'https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/{REVISION}/databricks-dolly-15k.jsonl'
 if not cache.exists():cache.write_bytes(urllib.request.urlopen(url,timeout=60).read())
 assert hashlib.sha256(cache.read_bytes()).hexdigest()==EXPECTED
 old=[]
 for path in (ROOT/'experiments').glob('*/**/*histories.jsonl'):
  for h in rows(path):
   old.extend(normalize(m['content']) for m in h.get('history',[]) if m['role']=='user')
 ban=re.compile(r'\b(json|bullet|list|format|paragraph|sentence|words?|code|python|sql|translate|exactly|responses?|answers?)\b',re.I)
 pool=[];seen=set()
 for idx,r in enumerate(rows(cache)):
  q=r['instruction'].strip();n=normalize(q)
  if r['category'] not in ['open_qa','brainstorming','creative_writing'] or r['context'].strip() or not 8<=len(q.split())<=70 or ban.search(q):continue
  if n in seen or any(n in t for t in old):continue
  seen.add(n);pool.append((idx,q,r['category']))
 random.Random(453).shuffle(pool)
 # Remove extremely similar questions without inspecting any generated answer.
 selected=[];bags=[]
 for row in pool:
  bag=set(re.findall(r'[a-z]+',row[1].casefold()))
  if any(len(bag&b)/max(1,len(bag|b))>.85 for b in bags):continue
  selected.append(row);bags.append(bag)
  if len(selected)==960:break
 assert len(selected)==960
 for i in range(240):
  qs=selected[4*i:4*i+4];name,args,rule,change=RULES[i%4]
  append(a.out/'candidates.jsonl',dict(case_id=f'confirm-{i:03d}',source_id=f'bundle-{i:03d}',candidate_rank=i,inst_name=name,inst_args=args,rule_text=rule+' Keep the entire reply at most 200 words.',change_text=change+' Keep this reply at most 200 words.',history_questions=[q[1] for q in qs[:3]],question=qs[3][1],provenance=dict(dataset='databricks/databricks-dolly-15k',revision=REVISION,rows=[q[0] for q in qs],categories=[q[2] for q in qs],response_field_used=False)))
 write(a.out/'source.json',dict(dataset='databricks/databricks-dolly-15k',revision=REVISION,url=url,sha256=EXPECTED,license='CC-BY-SA-3.0',attribution='Databricks, databricks-dolly-15k dataset',eligible_after_exact_filter=len(pool),candidate_histories=240,distinct_prompts=960,target_accepted=120,target_per_rule=30,candidate_per_rule=60,selection_seed=453,selection='Empty-context public questions only, 8..70 words, no listed competing format terms; exact dev-text and pairwise token-Jaccard >.85 exclusions. First 960 from seeded order. Four consecutive questions per bundle; round-robin rule assignment.',scope='MT-Eval rule/checker adaptation, not an official MT-Eval benchmark score. Bundles are constructed conversations, not original Dolly dialogues. No model outputs used in selection.'))
 (a.out/'LICENSE.md').write_text('Selected user questions are adapted from Databricks databricks-dolly-15k, licensed CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/ . Dataset: https://huggingface.co/datasets/databricks/databricks-dolly-15k . Original reference answers are not included or used.\n',encoding='utf-8',newline='\n')
 print({'candidates':240,'target_accepted':120,'unique_prompts':960,'new_model_calls':0})


if __name__=='__main__':main()
