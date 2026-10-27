import json
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from dashboard import arena_metrics as metrics


class ArenaMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.session = dict(mouse='VF093', assay='OFT', date='2026-08-17', session_dir=str(self.folder))
        columns = pd.MultiIndex.from_product(
            [['tracker'], ['neck', 'torso', 'tail_base'], ['x', 'y', 'likelihood']],
            names=['scorer', 'bodyparts', 'coords'])
        data = np.tile([500., 500., 1.], (100, 3))
        self.tracking = self.folder / 'remapped_cropped_test_clean.csv'
        pd.DataFrame(data, columns=columns).to_csv(self.tracking)
        pd.DataFrame({'frame': np.arange(100), 'label': ['Rearing'] * 10 + ['background'] * 90,
                      'score': .9, 'context_label': ['Rearing'] * 10 + ['background_center_static'] * 90
                      }).to_csv(self.folder / 'test_behavior.csv', index=False)

    def test_calculation_disk_reuse_and_source_invalidation(self):
        result = metrics.session_metrics(self.session)
        self.assertEqual(result['n_frames'], 100)
        self.assertEqual(result['OFT_ratio_time_center'], 1)
        self.assertEqual(result['OFT_distance_cm_per_min'], 0)
        self.assertAlmostEqual(result['OFT_rearing_time_ratio'], .1)
        self.assertEqual(result['fps_source'], 'fallback')
        saved = json.loads((self.folder / 'arena_metrics.json').read_text())
        self.assertIsNone(saved['metrics']['OFT_speed_cv'])
        with patch.object(metrics, 'compute_session_metrics', side_effect=AssertionError('Recomputed')):
            self.assertEqual(metrics.session_metrics(self.session), result)
        with self.tracking.open('a') as stream:
            stream.write('\n')
        with patch.object(metrics, 'compute_session_metrics', wraps=metrics.compute_session_metrics) as calculate:
            metrics.session_metrics(self.session)
            calculate.assert_called_once()

    def test_local_result_overrides_only_same_legacy_session(self):
        legacy = pd.DataFrame([dict(mouse='VF093', oft_date=pd.Timestamp('2026-08-17'), OFT_ratio_time_center=.2),
                               dict(mouse='VF093', oft_date=pd.Timestamp('2026-08-16'), OFT_ratio_time_center=.3)])
        frames, errors = metrics.collect_metrics(pd.DataFrame([self.session]), {'OFT': legacy})
        self.assertEqual(errors, [])
        self.assertEqual(len(frames['OFT']), 2)
        row = frames['OFT'].loc[frames['OFT']['oft_date'].eq(pd.Timestamp('2026-08-17'))].iloc[0]
        self.assertEqual(row['OFT_ratio_time_center'], 1)
        # Session detail must read the refreshed local frame, rather than stale SQLite.
        tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'openfield_session_metrics')
        namespace = {'pd': pd, 'OPENFIELD_ASSAY_CONFIG': {'OFT': {'table': 'oft_sessions', 'date': 'oft_date'}}}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'), namespace)
        _, selected = namespace['openfield_session_metrics']('VF093', 'OFT', '2026-08-17', frames)
        self.assertEqual(selected['OFT_ratio_time_center'], 1)

    def test_ldt_visibility_uses_reference_rules(self):
        session = {**self.session, 'assay': 'LDT'}
        tracking = pd.read_csv(self.tracking, header=[0, 1, 2], index_col=0)
        for column in tracking.columns:
            if column[2] == 'likelihood':
                tracking.loc[:49, column] = 0
        tracking.to_csv(self.tracking)
        result = metrics.session_metrics(session)
        self.assertEqual(result['LDT_dark_time_ratio'], .5)
        self.assertEqual(result['LDT_all_core_keypoints_visible_time_ratio'], .5)
        self.assertAlmostEqual(result['LDT_box_entries_per_min'], 30.)

    def test_unanalyzed_and_sit_sessions_are_not_fabricated(self):
        self.assertIsNone(metrics.session_metrics({**self.session, 'assay': 'SIT'}))
        self.tracking.unlink()
        self.assertIsNone(metrics.session_metrics(self.session))
