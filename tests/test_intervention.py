"""CPU checks of the intervention hook on a tiny random Qwen2 (no real weights)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'intervention'))
sys.path.insert(0, str(ROOT / 'reference_code'))

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:  # CPU report rebuilding does not need torch
    torch = None


@unittest.skipIf(torch is None, 'torch/transformers not installed')
class HookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=512)
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.ids = [int(x) for x in torch.randint(3, 97, (23,))]
        cls.vec = torch.randn(32) * 3

    def logits(self, ids, steer_vec=None, start=None):
        from steer import Steer
        with torch.inference_mode(), Steer(self.model, 2, steer_vec) as st:
            st.start = start
            return self.model(input_ids=torch.tensor([ids]), output_hidden_states=True)

    def test_zero_vector_is_bitwise_identity(self):
        a = self.logits(self.ids).logits
        b = self.logits(self.ids, torch.zeros(32), len(self.ids) - 1).logits
        self.assertTrue(torch.equal(a, b))

    def block_outputs(self, vec, start):
        """Outputs of blocks 2 and 4, read by hooks registered after Steer (output_hidden_states
        records layer inputs in transformers 5, so it cannot see the edit)."""
        from steer import Steer
        seen = {}
        def rec(i):
            def h(mod, args, out):
                seen[i] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return h
        with torch.inference_mode(), Steer(self.model, 2, vec) as st:
            hs = [self.model.model.layers[i - 1].register_forward_hook(rec(i)) for i in (2, 4)]
            st.start = start
            self.model(input_ids=torch.tensor([self.ids]))
            for h in hs:
                h.remove()
        return seen

    def test_only_scoped_positions_change(self):
        start = len(self.ids) - 1
        base, hit = self.block_outputs(None, None), self.block_outputs(self.vec, start)
        self.assertTrue(torch.equal(base[2][0, :start], hit[2][0, :start]))
        self.assertTrue(torch.allclose(hit[2][0, start] - base[2][0, start], self.vec, atol=1e-5))
        self.assertTrue(torch.equal(base[4][0, :start], hit[4][0, :start]))  # causal: earlier positions stay identical
        self.assertFalse(torch.equal(base[4][0, start], hit[4][0, start]))

    def test_decode_and_teacher_forced_scopes_agree(self):
        from steer import Steer, generate, teacher_forced_logprobs
        with Steer(self.model, 2, self.vec) as st:
            g = generate(self.model, self.ids, eos={0}, cap=12, steer=st)
        self.assertEqual(st.calls[0], (len(self.ids), len(self.ids) - 1))
        self.assertTrue(all(c == (1, 0) for c in st.calls[1:]))
        with Steer(self.model, 2, self.vec) as st:
            logp = teacher_forced_logprobs(self.model, self.ids, g['generated_ids'], st)
        self.assertEqual(logp.argmax(-1).tolist(), g['generated_ids'])
        with Steer(self.model, 2, None) as st:
            plain = generate(self.model, self.ids, eos={0}, cap=12, steer=st)
        self.assertNotEqual(plain['generated_ids'], g['generated_ids'])

    def test_kl_floor_and_positive(self):
        from steer import Steer, kl_summary, teacher_forced_logprobs
        ref = [5, 9, 11, 2]
        with Steer(self.model, 2, None) as st:
            a = teacher_forced_logprobs(self.model, self.ids, ref, st)
        with Steer(self.model, 2, self.vec) as st:
            b = teacher_forced_logprobs(self.model, self.ids, ref, st)
        self.assertEqual(kl_summary(a, a, ref)['kl_mean'], 0.0)
        self.assertGreater(kl_summary(a, b, ref)['kl_mean'], 0.0)


@unittest.skipIf(torch is None, 'torch/transformers not installed')
class SerializationTests(unittest.TestCase):
    def test_b_and_c_match_original_engine_messages(self):
        import migration_exploration_core_v1 as core
        from run import messages
        history = [dict(role='user', content='rule\n\nq1'), dict(role='assistant', content='a1 ****** a2'),
                   dict(role='user', content='q2'), dict(role='assistant', content='b1')]
        self.assertEqual(messages(history, 'now?', 'B'), core.messages(history, 'now?', pasted=False))
        self.assertEqual(messages(history, 'now?', 'C'), core.messages(history, 'now?', pasted=True))
        self.assertIn('Q:\nrule', messages(history, 'now?', 'C_qa')[1]['content'])
        self.assertIn('"role": "assistant"', messages(history, 'now?', 'C_json')[1]['content'])


if __name__ == '__main__':
    unittest.main()
