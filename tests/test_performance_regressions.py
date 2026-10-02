"""Run with: python -m unittest discover -s tests"""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.ndimage import gaussian_filter1d

SOURCE = Path(__file__).resolve().parents[1] / 'app.py'

def functions(*names):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(np=np, pd=pd, feather=feather, Path=Path, json=json,
                     gaussian_filter1d=gaussian_filter1d)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace

class PerformanceRegressionTests(unittest.TestCase):
    def test_replay_selects_exact_row_and_version_without_changing_values(self):
        ns = functions('load_behavior_replay_table', 'load_behavior_video_session')
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_behavior_replay_table')
        columns = ast.literal_eval(helper.body[1].value)
        frame = pd.DataFrame({column: [np.array([1., 2.]), np.array([3., 4.])] for column in columns})
        frame['Mouse_ID'] = ['mouse', 'mouse']
        frame['Date'] = ['2026-01-01', '2026-01-01']
        frame['Version'] = [1, 2]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'behavior.feather')
            frame.to_feather(path)
            load = ns['load_behavior_video_session']
            for version, index, expected in [('1', 0, [1., 2.]), ('2', None, [3., 4.]), ('2', 0, [3., 4.])]:
                row = load(path, (), 'mouse', '2026-01-01', version, index)
                np.testing.assert_array_equal(row['Position'], expected)
                self.assertEqual(str(row['Version']), version)
            self.assertIsNone(load(path, (), 'other', '2026-01-01', '1'))

    def test_vectorized_bins_match_indexed_accumulation(self):
        rng = np.random.default_rng(42)
        rows = rng.integers(0, 384, 100_000)
        bins = rng.integers(0, 300, len(rows))
        expected = np.zeros((384, 300))
        np.add.at(expected, (rows, bins), 1)
        actual = np.bincount(rows * 300 + bins, minlength=expected.size).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_analysis_preserves_counts_and_activity(self):
        ns = functions('casefold_child', 'casefold_path', 'load_probe_analysis')
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            probe = session / 'alf' / 'probe00'
            labels = session / 'bombcell' / 'probe00'
            probe.mkdir(parents=True); labels.mkdir(parents=True)
            locations = probe / 'channel_locations.json'
            locations.write_text(json.dumps({f'channel_{i}': {'axial': i*20, 'brain_region': 'MOs'} for i in range(12)}))
            pd.DataFrame({'bombcell_label': ['good'] * 12}).to_csv(labels / 'bombcell_labels.csv')
            np.save(probe / 'clusters.channels.npy', np.arange(12))
            # Cross the chunk boundary and include out-of-range clusters.
            clusters = np.arange(1_000_019) % 14 - 1
            times = np.linspace(0, 100, len(clusters))
            np.save(probe / 'spikes.times.npy', times)
            np.save(probe / 'spikes.clusters.npy', clusters)
            result = ns['load_probe_analysis'](str(locations), (), on_site_arrival_times=[20., 40.])
            expected = np.bincount(clusters[(clusters >= 0) & (clusters < 12)], minlength=12)
            np.testing.assert_array_equal(result['units']['num_spikes'], expected)
            np.testing.assert_allclose(result['units']['firing_rate'], expected/100)
            self.assertIsNotNone(result['activity_heatmap'])
            self.assertIsNotNone(result['regional_activity'])
            self.assertEqual(result['peri_bout']['arrival_count'], 2)

if __name__ == '__main__':
    unittest.main()
