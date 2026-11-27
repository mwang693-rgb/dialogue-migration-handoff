"""CPU checks of the transfer unit builder and summary statistics."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RECOVERY = ROOT.parent / 'dialogue-migration-recovery' / 'scripts'


def load(name):
    """Load transfer/<name>.py under a unique module name (intervention/ has files with the same names)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f'transfer_{name}', ROOT / 'transfer' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


summarize = load('summarize')
cluster_boot, decide, idx = summarize.cluster_boot, summarize.decide, summarize.idx


class StatsTests(unittest.TestCase):
    def test_constant_values_give_degenerate_interval(self):
        r = cluster_boot([0.3] * 6, ['a', 'a', 'b', 'c', 'c', 'c'], repeats=200)
        self.assertAlmostEqual(r['point'], 0.3)
        self.assertAlmostEqual(r['ci'][0], 0.3)
        self.assertEqual(r['n_histories'], 3)

    def test_clusters_are_resampled_whole(self):
        # cluster a is all +1, cluster b all -1; any draw mean is a mixture of whole clusters
        r = cluster_boot([1, 1, 1, -1], ['a', 'a', 'a', 'b'], repeats=500)
        self.assertGreaterEqual(r['ci'][0], -1)
        self.assertLessEqual(r['ci'][1], 1)
        self.assertAlmostEqual(r['point'], 0.5)

    def test_index_is_bounded_and_zero_safe(self):
        self.assertEqual(idx(0.0, 0.0), 0.0)
        self.assertAlmostEqual(idx(3.0, 1.0), 0.5)
        self.assertAlmostEqual(idx(1.0, 3.0), -0.5)

    def test_decisions(self):
        x = lambda lo, hi: dict(point=(lo + hi) / 2, ci=[lo, hi])
        yes = dict(P1=dict(L_sign=x(0.1, 0.3), L_random=x(0.01, 0.2)), P2=dict(S_random=x(0.02, 0.1), S_sign=x(0.1, 0.2)))
        self.assertEqual(decide(yes), ('length_effect_transfers', 'kl_specificity_transfers'))
        null = dict(P1=dict(L_sign=x(-0.04, 0.04), L_random=x(-0.2, 0.2)), P2=dict(S_random=x(-0.03, 0.03), S_sign=x(-0.01, 0.04)))
        self.assertEqual(decide(null), ('no_length_effect', 'no_kl_specificity'))
        mixed = dict(P1=dict(L_sign=x(0.1, 0.3), L_random=x(-0.01, 0.2)), P2=dict(S_random=x(0.02, 0.1), S_sign=x(-0.1, 0.2)))
        self.assertEqual(decide(mixed), ('inconclusive', 'inconclusive'))


@unittest.skipUnless((RECOVERY / 'run_pilot.py').exists(), 'sibling recovery repo not present')
class SerializerTests(unittest.TestCase):
    def test_pilot_copy_matches_original(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('run_pilot_original', RECOVERY / 'run_pilot.py')
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(RECOVERY))  # run_pilot imports its sibling pilot_data
        try:
            spec.loader.exec_module(mod)
        except ImportError as e:
            self.skipTest(f'run_pilot dependencies missing: {e}')
        pilot_messages = load('build_units').pilot_messages
        history = [dict(role='user', content='u1'), dict(role='assistant', content='a1'), dict(role='user', content='u2'),
                   dict(role='assistant', content='a2')]
        for pasted in (False, True):
            self.assertEqual(pilot_messages(history, 'q?', pasted), mod.messages(history, 'q?', pasted=pasted))


if __name__ == '__main__':
    unittest.main()
