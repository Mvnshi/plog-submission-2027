"""Executable metric contracts, using only constructed synthetic label sequences.

These tests neither load the competition's public GT nor select a PLOG model.
They preserve surprising official behavior so engineering assumptions cannot
silently replace the actual competition scorer.
"""

import unittest

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm


def codes(text):
    return pm.label_codes(list(text))


class OfficialMetricTests(unittest.TestCase):
    def test_spell_thresholds_and_inclusive_exact_iou(self):
        self.assertEqual(pm.SPELL_IOUS, (.3, .4, .5, .6, .7))
        gt, pred = codes('H' * 10), codes('H' * 5 + 'D' * 5)
        result = pm.evaluate(gt, pred)
        # One GT spell and one predicted spell with exactly 5/10 temporal IoU.
        for threshold in (.3, .4, .5):
            self.assertEqual(result['spell_f1'][threshold]['n_matched'], 1)
        for threshold in (.6, .7):
            self.assertEqual(result['spell_f1'][threshold]['n_matched'], 0)
        self.assertAlmostEqual(result['spell_f1_mean'], .6)

    def test_spell_overlaps_same_team_only_and_stable_sort(self):
        gt = [(0, 10, pm.H), (20, 30, pm.H), (40, 50, pm.A)]
        pred = [(0, 30, pm.H), (0, 10, pm.H), (40, 50, pm.H)]
        overlaps = pm.spell_overlaps(gt, pred)
        self.assertEqual(overlaps[0], (1.0, 0, 1))
        self.assertEqual(overlaps[1:], [(1 / 3, 0, 0), (1 / 3, 1, 0)])
        # An identical interval with the opposite team gets no overlap candidate.
        self.assertFalse(any(gt_index == 2 for _, gt_index, _ in overlaps))

    def test_greedy_matching_consumes_each_side_once(self):
        # Direct candidates expose greedy semantics: this is not a maximum-cardinality
        # bipartite assignment. The highest-IoU edge consumes GT0 and prediction0.
        overlaps = [(0.9, 0, 0), (0.8, 0, 1), (0.8, 1, 0)]
        self.assertEqual(pm.match_spells(overlaps, 2, 2, .3), (1, 1, 1))
        np.testing.assert_array_equal(pm.matched_gt_flags(overlaps, 2, .3), [True, False])
        # A prediction split in two must not earn two credits for one true spell.
        gt, pred = codes('H' * 10), codes('H' * 4 + 'A' * 2 + 'H' * 4)
        result = pm.evaluate(gt, pred)
        self.assertEqual(result['spell_f1'][.3]['n_matched'], 1)
        self.assertEqual(result['n_pred_spells'], 3)

    def test_dead_ball_separates_same_team_spells(self):
        gt = codes('HHDDHH')
        spells = pm.possession_spells(gt, pm.valid_runs(gt))
        self.assertEqual(spells, [(0, 2, pm.H), (4, 6, pm.H)])
        self.assertEqual(pm.evaluate(gt, gt)['spell_f1_mean'], 1.0)

    def test_abstentions_forward_then_head_backfill_only_inside_gt_runs(self):
        gt = codes('XXHHHHXXAAAAXX')
        pred = codes('AAXXHXAAXXAXHH')
        runs = pm.valid_runs(gt)
        self.assertEqual(runs, [(2, 6), (8, 12)])
        filled = pm.fill_abstentions(pred, runs)
        np.testing.assert_array_equal(filled, codes('AAHHHHA AAAAAHH'.replace(' ', '')))
        # The first run's leading X is H, not the A immediately before the run.
        self.assertEqual(filled[2], pm.H)
        self.assertEqual(filled[8], pm.A)
        # Ungraded predictions remain unchanged, and caller input is never mutated.
        np.testing.assert_array_equal(pred, codes('AAXXHXAAXXAXHH'))

    def test_all_x_run_stays_x_and_does_not_inherit_other_half(self):
        gt, pred = codes('HHXXAA'), codes('HHXXXX')
        filled = pm.fill_abstentions(pred, pm.valid_runs(gt))
        np.testing.assert_array_equal(filled, pred)
        self.assertEqual(pm.possession_spells(filled, pm.valid_runs(gt)), [(0, 2, pm.H)])
        self.assertEqual(pm.transitions(filled, pm.valid_runs(gt)), [])

    def test_x_costs_frame_f1_but_can_be_free_for_spell_f1(self):
        gt, pred = codes('HHHHAAAADDDD'), codes('HXXHAXXDDDDD')
        # Keep the true D boundary at frame8; abstentions only inside live spells.
        pred[7] = pm.A
        result = pm.evaluate(gt, pred)
        self.assertEqual(result['spell_f1_mean'], 1.0)
        self.assertLess(result['macro_f1'], 1.0)
        self.assertEqual(result['n_abstain_on_graded'], 4)

    def test_x_at_handover_carries_old_team_and_costs_spells(self):
        gt = codes('H' * 10 + 'A' * 10 + 'D' * 10)
        pred = codes('H' * 10 + 'X' * 9 + 'A' + 'D' * 10)
        filled = pm.fill_abstentions(pred, pm.valid_runs(gt))
        np.testing.assert_array_equal(filled[10:19], pm.H)
        self.assertLess(pm.evaluate(gt, pred)['spell_f1_mean'], 1.0)

    def test_absent_class_is_zero_and_ungraded_false_positives_are_ignored(self):
        gt, pred = codes('HHXX'), codes('HHAA')
        result = pm.evaluate(gt, pred)
        self.assertEqual(result['per_class']['A']['f1'], 0.0)
        self.assertEqual(result['per_class']['D']['f1'], 0.0)
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['n_graded'], 2)
        self.assertEqual(result['spell_f1_mean'], 1.0)

    def test_invalid_label_and_blank_are_not_silent_abstentions(self):
        with self.assertRaises(ValueError):
            pm.label_codes(['H', '?'])
        for labels in (['H', None], ['H', ''], ['H', 'Q']):
            table = pd.DataFrame({'frame': [0, 1], 'label': labels})
            self.assertIsNotNone(pm.validate_submission(table, 2))

    def test_one_frame_legitimate_possession_has_full_metric_weight(self):
        gt = codes('H' * 100 + 'A' + 'H' * 100 + 'D' * 10)
        result = pm.evaluate(gt, gt)
        self.assertEqual(result['n_gt_spells'], 3)
        self.assertEqual(result['spell_f1_mean'], 1.0)
        oversmoothed = codes('H' * 201 + 'D' * 10)
        self.assertLess(pm.evaluate(gt, oversmoothed)['spell_f1_mean'], .3)

    def test_high_frame_accuracy_can_coexist_with_zero_spell_f1(self):
        gt = codes('H' * 1000 + 'A' * 1000 + 'D' * 1000)
        pred = gt.copy()
        pred[10:1000:20] = pm.A
        pred[1010:2000:20] = pm.H
        result = pm.evaluate(gt, pred)
        self.assertGreater(result['accuracy'], .96)
        self.assertGreater(result['macro_f1'], .96)
        self.assertEqual(result['spell_f1_mean'], 0.0)
        self.assertGreater(result['n_pred_spells'], 100)

    def test_constant_guess_can_score_on_constructed_short_ground_truth(self):
        # EVALUATION's constant=0 claim describes its benchmark, not all inputs.
        gt = codes('H' * 10)
        result = pm.evaluate(gt, np.full(10, pm.H, dtype=np.int8))
        self.assertEqual(result['spell_f1_mean'], 1.0)
        # D and X have no predicted team spells, so those constants do score zero.
        for label in (pm.D, pm.X):
            self.assertEqual(pm.evaluate(gt, np.full(10, label, dtype=np.int8))['spell_f1_mean'], 0.0)

    def test_transition_destination_matching_is_one_to_one(self):
        gt = [(100, pm.D, pm.H), (200, pm.H, pm.A)]
        pred = [(99, pm.A, pm.H), (101, pm.D, pm.H), (201, pm.H, pm.A)]
        pairs, missed, spurious = pm.match_transitions(gt, pred, 2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(missed, [])
        self.assertEqual(spurious, [pred[1]])
        # Origin differs on the first match; requiring full from/to rejects it.
        pair_typed = pm.match_transitions(gt, [pred[0]], 2, typed='pair')
        self.assertEqual(len(pair_typed[0]), 0)


if __name__ == '__main__':
    unittest.main()
