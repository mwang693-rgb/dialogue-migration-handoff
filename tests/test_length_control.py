"""CPU checks for the length-direction control."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOKENIZER = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'


def load(name):
    spec = importlib.util.spec_from_file_location(f'length_control_{name}', ROOT / 'length_control' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DecisionTests(unittest.TestCase):
    def test_decisions(self):
        decide = load('summarize').decide
        x = lambda lo, hi: dict(point=(lo + hi) / 2, ci=[lo, hi])
        self.assertEqual(decide(dict(M=x(-0.1, 0.2), L_perp=x(0.1, 0.2), S_len=x(0.1, 0.2))), 'invalid_length_control')
        self.assertEqual(decide(dict(M=x(0.1, 0.2), L_perp=x(0.1, 0.2), S_len=x(0.01, 0.2))), 'not_just_length')
        self.assertEqual(decide(dict(M=x(0.1, 0.2), L_perp=x(-0.04, 0.04), S_len=x(-0.2, 0.03))), 'mostly_length')
        self.assertEqual(decide(dict(M=x(0.1, 0.2), L_perp=x(0.1, 0.2), S_len=x(-0.1, 0.2))), 'mixed')

    def test_vector_decomposition(self):
        import torch
        v = load('run').vectors(torch.tensor([3.0, 4.0]), torch.tensor([2.0, 0.0]))
        self.assertTrue(torch.allclose(v['d_par'], torch.tensor([3.0, 0.0])))
        self.assertTrue(torch.allclose(v['d_perp'], torch.tensor([0.0, 4.0])))
        self.assertAlmostEqual(float(v['len'].norm()), 5.0, places=5)


@unittest.skipUnless(TOKENIZER.exists(), 'local Qwen tokenizer not present')
class InsertionTests(unittest.TestCase):
    def test_instruction_goes_before_final_im_end(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
        msgs = [dict(role='system', content='You are a helpful assistant.'), dict(role='user', content='Tell me about tea.')]
        base = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)['input_ids']
        out = load('run').with_instruction(tok, base, '\n\nPlease keep your answer brief.')
        text = tok.decode(out)
        self.assertTrue(text.endswith('Tell me about tea.\n\nPlease keep your answer brief.<|im_end|>\n<|im_start|>assistant\n'))


if __name__ == '__main__':
    unittest.main()
