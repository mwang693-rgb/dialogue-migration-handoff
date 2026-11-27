"""CPU checks for the multi-model replication helpers."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(f'replication_{name}', ROOT / 'replication' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EngineHelperTests(unittest.TestCase):
    def test_depths_match_qwen_layers(self):
        d = load('engine').depth_layers
        self.assertEqual(d(28), [0, 4, 8, 12, 16, 20, 24, 28])
        self.assertEqual(d(32), [0, 5, 9, 14, 18, 23, 27, 32])
        self.assertEqual(d(42), [0, 6, 12, 18, 24, 30, 36, 42])

    def test_messages_forms(self):
        m = load('engine').messages
        h = [dict(role='user', content='u1'), dict(role='assistant', content='a1')]
        b = m(h, 'q?', 'B', True)
        self.assertEqual([x['role'] for x in b], ['system', 'user', 'assistant', 'user'])
        c = m(h, 'q?', 'C', False, 'pilot')
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]['content'], 'User:\nu1\n\nAssistant:\na1\n\nLatest user message:\nq?')
        self.assertTrue(m(h, 'q?', 'C', True)[1]['content'].endswith('--- Current user request ---\nq?'))
        self.assertIn('Q:\nu1', m(h, 'q?', 'C_qa', True)[1]['content'])

    def test_matches_intervention_serializer(self):
        try:
            import torch  # noqa: F401  (intervention/run.py imports it lazily only)
        except ImportError:
            self.skipTest('torch missing')
        sys.path.insert(0, str(ROOT / 'intervention'))
        spec = importlib.util.spec_from_file_location('intervention_run', ROOT / 'intervention' / 'run.py')
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        m = load('engine').messages
        h = [dict(role='user', content='rule\n\nq1'), dict(role='assistant', content='a1 ****** a2')]
        for form in ('B', 'C', 'C_qa', 'C_json'):
            self.assertEqual(m(h, 'now?', form, True), run.messages(h, 'now?', form))

    def test_decide(self):
        decide = load('summarize').decide
        x = lambda lo, hi: dict(point=(lo + hi) / 2, ci=[lo, hi])
        self.assertEqual(decide(dict(L_sign=x(0.1, 0.3), L_random=x(0.01, 0.2))), 'replicates')
        self.assertEqual(decide(dict(L_sign=x(-0.04, 0.04), L_random=x(-0.2, 0.2))), 'absent')
        self.assertEqual(decide(dict(L_sign=x(0.1, 0.3), L_random=x(-0.01, 0.2))), 'inconclusive')


if __name__ == '__main__':
    unittest.main()
