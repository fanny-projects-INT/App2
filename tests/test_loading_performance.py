"""Numerical equivalence and cache isolation for dashboard loading."""
import ast
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import streamlit as st
from dashboard.app_functions import compute_failures, count_reward_per_bout, flatten_nested_times, prepare_mouse_dataframe, count_valid_bouts

ROOT = Path(__file__).resolve().parents[1]

def reference(row):
    failures, rewards = [], []
    ts, bouts = row['Timestamps'], row['Bout for Timestamps']
    if not len(ts) or not len(bouts): return failures, rewards
    rewarded = flatten_nested_times(row['Times Rewarded Licks'])
    non_rewarded = flatten_nested_times(row['Times Non Rewarded Licks'])
    for bout in np.unique(bouts):
        times = ts[bouts == bout]
        r = [t for t in rewarded if len(times) and times[0] <= t <= times[-1]]
        nr = [t for t in non_rewarded if len(times) and times[0] <= t <= times[-1]]
        rewards.append(len(r))
        failures.append(sum(t > max(r) for t in nr) if r else len(nr))
    return failures, rewards

class LoadingPerformanceTests(unittest.TestCase):
    def test_counts_match_original_with_unsorted_and_missing_events(self):
        rng = np.random.default_rng(44)
        for _ in range(60):
            n = int(rng.integers(0, 300))
            ts = rng.integers(0, 100, n).astype(float)
            bouts = rng.integers(0, 15, n).astype(float)
            if n:
                ts[rng.integers(n)] = np.nan
                bouts[rng.integers(n)] = np.nan
            row = {'Timestamps': ts, 'Bout for Timestamps': bouts,
                   'Times Rewarded Licks': [rng.integers(0, 100, 50), np.array([np.nan])],
                   'Times Non Rewarded Licks': [rng.integers(0, 100, 90), np.array([np.nan])]}
            expected = reference(row)
            self.assertEqual(compute_failures(row), expected[0])
            self.assertEqual(count_reward_per_bout(row), expected[1])

    def test_mouse_order_latest_first_with_stable_ties_and_missing_dates(self):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mice_by_latest_session')
        ns = dict(pd=pd, is_visible_mouse=lambda mouse: mouse != 'VF000')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'order', 'exec'), ns)
        frame = pd.DataFrame({'mouse':['B','A','B','C','VF000','D'],
                              'date':['2025-01-01','2026-01-01','2026-01-01',None,'2027-01-01','2024-01-01']})
        self.assertEqual(ns['mice_by_latest_session'](frame, 'mouse', 'date'), ['A','B','D','C'])
        self.assertEqual(ns['mice_by_latest_session'](pd.DataFrame()), [])

    def test_mouse_cache_reuses_data_and_invalidates_on_file_change(self):
        names = {'load_behavior_catalog', 'read_behavior_rows', 'load_behavior_mouse',
                 'load_recent_behavior_mice', 'load_older_behavior_mouse',
                 'prepare_cached_behavior_frame', 'mice_by_latest_session'}
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = dict(st=st, pd=pd, pa=pa, feather=feather, prepare_mouse_dataframe=prepare_mouse_dataframe,
                  count_valid_bouts=count_valid_bouts, is_visible_mouse=lambda mouse: mouse not in {"VF000", "VF001"})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/'app.py'), 'exec'), ns)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'data.feather'
            frame = pd.DataFrame({'Mouse_ID':['B','A','B'], 'Date':['2026-02-02','2026-01-01','2026-01-01'],
                'Version':[2,1,1], 'Protocol':[1,2,3], 'Probas':['90/30']*3,
                'Correct Bouts':[[True],[True,False],[False]], 'Timestamps':[[9.],[1.],[3.]]})
            older = pd.DataFrame([dict(Mouse_ID=f"old-{i:02}", Date="2020-01-01", Version=1,
                                      Protocol=1, Probas="90/30", **{"Correct Bouts":[True], "Timestamps":[float(i)]})
                                  for i in range(20)])
            frame = pd.concat([frame, older], ignore_index=True)
            frame.to_feather(p)
            load = ns['load_behavior_mouse']
            try:
                with patch.object(pd, 'read_feather', wraps=pd.read_feather) as read:
                    catalog = ns['load_behavior_catalog'](str(p), (1,))
                    self.assertEqual(read.call_args.kwargs['columns'], ['Mouse_ID','Date','Version','Protocol'])
                    self.assertNotIn('Timestamps', catalog)
                first = load(str(p), (1,), 'B')
                self.assertIs(first, load(str(p), (1,), 'B'))
                self.assertEqual(first['Mouse_ID'].tolist(), ['B','B'])
                self.assertEqual([a.tolist() for a in first['Timestamps']], [[3.],[9.]])
                # More than eight older visits must not evict the recent group.
                for i in range(10, 20):
                    load(str(p), (1,), f"old-{i:02}")
                self.assertIs(first, load(str(p), (1,), 'B'))
                private = first.copy(); private['label'] = 'changed'
                self.assertNotIn('label', load(str(p), (1,), 'B'))
                frame.loc[0,'Version'] = 3; frame.to_feather(p)
                updated = load(str(p), (2,), 'B')
                self.assertIsNot(updated, first)
                self.assertEqual(updated['Version'].tolist(), ['1','3'])
            finally:
                for name in names:
                    if hasattr(ns[name], "clear"): ns[name].clear()

if __name__ == '__main__': unittest.main()
