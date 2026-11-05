import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff import analyze_all, check_edit, features, rows, check_strategy


class MetricPitfalls(unittest.TestCase):
    def test_whitespace_is_not_an_edit_failure(self):
        h = rows('experiments/01_editing/histories.jsonl')[0]
        q = h['queries'][0]
        self.assertTrue(check_edit('```json\n'+json.dumps(q['expected'], indent=4)+'\n```','book',q)['all_pass'])

    def test_preserving_everything_does_not_mean_target_edit_succeeded(self):
        h = rows('experiments/01_editing/histories.jsonl')[0]
        q = h['queries'][0]
        result = check_edit(json.dumps(q['original']), 'book', q)
        self.assertEqual(result['retained_unchanged_items'], 7)
        self.assertFalse(result['all_pass'])

    def test_retention_membership_does_not_check_order(self):
        h = rows('experiments/01_editing/histories.jsonl')[0]
        q = h['queries'][1]
        result = check_edit(json.dumps(q['original']), 'book', q)
        self.assertEqual(result['retained_unchanged_items'], 8)
        self.assertFalse(result['all_pass'])

    def test_plain_and_curly_apostrophes(self):
        self.assertEqual(features("We’re ready. Don't stop; do not stop.")['count'], 2)

    def test_ambiguous_apostrophe_s_is_not_counted(self):
        self.assertEqual(features("It's John's book, and he's here.")['count'], 0)

    def test_two_responses_checker_is_a_surface_proxy(self):
        h = next(h for h in rows('experiments/02_strategy/histories.jsonl') if h['inst_name']=='combination:two_responses')
        # The source checker checks separated responses, not semantic difference.
        self.assertTrue(check_strategy('The answer is yes. ****** Yes is the answer.', h))
        self.assertFalse(check_strategy('Same text. ****** Same text.', h))
        self.assertFalse(check_strategy('One response only.', h))

    def test_entire_export_recomputes_and_new_gate_remains_failed(self):
        result = analyze_all()
        self.assertEqual(result['editing']['metrics']['P']['joint_correct'], 17)
        self.assertEqual(result['strategy']['extension']['methods']['CPasted']['passed'], 9)
        self.assertFalse(result['style']['frozen_gate_pass'])
        self.assertEqual(result['style']['steering_outputs'], 0)


if __name__ == '__main__':
    unittest.main()
