import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from confirmation.common import ROOT,read,check_rule,split_sources,request_text,validate_cases
from confirmation.score import validate_old,bootstrap_ci


class ConfirmationProtocol(unittest.TestCase):
    def test_override_reminder_repeats_new_rule(self):
        cfg=read(ROOT/'confirmation/protocol.json')
        ordinary=request_text('Explain a bicycle.','ordinary','C_reminder',cfg)
        change=request_text('Explain a bicycle.','change','C_reminder',cfg)
        self.assertTrue(ordinary.endswith(cfg['ordinary_rule']))
        self.assertTrue(change.endswith(cfg['override']))
        self.assertNotIn(cfg['ordinary_rule'],change)

    def test_rule_checker_obeys_current_request(self):
        two='Version one. ****** Version two.'
        self.assertTrue(check_rule(two,'ordinary')['passed'])
        self.assertFalse(check_rule(two,'change')['passed'])
        self.assertTrue(check_rule('One answer.','change')['passed'])
        self.assertFalse(check_rule(two,'ordinary',True)['passed'])
        self.assertFalse(check_rule('Same. ****** Same.','ordinary')['passed'])
        self.assertFalse(check_rule('a ****** '+'word '*200,'ordinary')['passed'])

    def test_source_split_is_stable_and_grouped(self):
        ids=[f's{i}' for i in range(40)]
        splits=split_sources(ids)
        self.assertEqual(splits,split_sources(ids*3))
        self.assertEqual(set(splits.values()),{'train','validation','test'})

    def test_old_eighty_outputs_still_score_as_recorded(self):
        self.assertEqual(validate_old()['old_records_verified'],80)

    def test_bootstrap_resamples_paired_source_differences(self):
        self.assertEqual(bootstrap_ci([1]*12,repeats=100),[1,1])
        self.assertEqual(bootstrap_ci([0]*12,repeats=100),[0,0])

    def test_duplicate_questions_and_unsafe_ids_are_rejected(self):
        cfg=read(ROOT/'confirmation/protocol.json')
        c=dict(case_id='source-1',source_id='source-1',history_questions=['one','two','three'],question='four',provenance='fixture')
        validate_cases([c],cfg)
        c['case_id']='../outside'
        with self.assertRaises(AssertionError): validate_cases([c],cfg)
        c['case_id']='source-1';c['question']='one'
        with self.assertRaises(AssertionError):validate_cases([c],cfg)

    def test_sample_counts_match_four_conditions(self):
        cfg=read(ROOT/'confirmation/protocol.json');plan=cfg['sample_plan']
        self.assertEqual(plan['histories']*plan['requests_per_history']*len(cfg['conditions']),plan['final_outputs'])
        self.assertEqual(plan['histories']*plan['change_requests_per_history']*len(cfg['conditions']),plan['change_outputs'])
        self.assertEqual(plan['histories']*plan['requests_per_history']*2*len(cfg['layers']),plan['activation_records'])

    def test_v2_candidate_pool_and_current_rule_checks(self):
        from confirmation.common import rows
        cfg=read(ROOT/'confirmation/protocol.json')
        cases=rows(ROOT/'data/confirmation_v2/candidates.jsonl')
        validate_cases(cases,cfg,confirmation=True)
        from collections import Counter
        self.assertEqual(set(Counter(c['inst_name'] for c in cases).values()),{60})
        forms={'combination:two_responses':'First. ****** Second.',
               'format:json_format':'{"answer": "test"}',
               'format:bullet_list':'- First\n- Second',
               'startend:end_phrase':'Test. Is there anything else I can help with?'}
        for c in cases[:4]:
            old=forms[c['inst_name']]
            self.assertTrue(check_rule(old,'ordinary',False,200,c['inst_name'],c['inst_args'])['passed'])
            self.assertFalse(check_rule(old,'change',False,200,c['inst_name'],c['inst_args'])['passed'])
            self.assertTrue(check_rule('A plain reply.','change',False,200,c['inst_name'],c['inst_args'])['passed'])
            self.assertTrue(request_text('Question','change','C_reminder',cfg,c).endswith(c['change_text']))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'PyTorch needed for tensor interface tests')
class ActivationExport(unittest.TestCase):
    def test_eight_layer_capture_reads_last_prompt_position(self):
        import torch
        from types import SimpleNamespace
        from confirmation.run_batch import capture_feed
        torch.set_num_threads(1)
        network=SimpleNamespace(embed_tokens=torch.nn.Embedding(256,12),layers=[torch.nn.Linear(12,12) for _ in range(28)])
        class Engine:
            model=SimpleNamespace(model=network)
            def feed(self,ids,cache):
                with torch.no_grad():
                    h=network.embed_tokens(torch.tensor([ids]))
                    for layer in network.layers:h=layer(h)
                return h[0,-1],cache
        layers=[0,4,8,12,16,20,24,28]
        _,_,states=capture_feed(Engine(),[1,2,198],None,layers)
        self.assertEqual(set(states),set(layers))
        self.assertTrue(torch.equal(states[0],network.embed_tokens.weight[198]))
        self.assertTrue(all(s.shape==(12,) and s.dtype==torch.float32 and s.device.type=='cpu' for s in states.values()))
        self.assertTrue(all(not m._forward_hooks for m in [network.embed_tokens,*network.layers]))

    def test_sishen_contract_and_corruption_check(self):
        import torch
        from confirmation.make_fake_data import make_fake
        from confirmation.validate import validate_run
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'fake'
            result=make_fake(folder,sources=2,hidden_size=3584)
            self.assertEqual(result['outputs'],16)
            self.assertEqual(result['activation_records'],64)
            self.assertEqual(result['hidden_size'],3584)
            with self.assertRaises(AssertionError):validate_run(folder,require_real=True)
            f=next((folder/'activations').glob('*.pt'))
            payload=torch.load(f,weights_only=True)
            payload['records'][0]['n_tokens']+=1
            torch.save(payload,f)
            with self.assertRaises(AssertionError):validate_run(folder)


if __name__=='__main__':unittest.main()
