import ast
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class BehaviorTabsTests(unittest.TestCase):
    def test_only_active_tab_builds_foreground_plots(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))
        render = ast.unparse(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                 and node.name == "render_behavior_view"))
        callback = ast.unparse(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                   and node.name == "remember_behavior_session"))
        fixtures = '''
import streamlit as st
from dashboard.ui_states import dashboard_tabs
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import contextmanager
BEHAVIOR_DB=Path('app.py')
PROTOCOL_LABELS={1:'Training 1'}
st.session_state.setdefault('plot_calls', 0)
st.session_state.setdefault('metric_calls', 0)
def source_signature(path): return 'test'
def load_behavior_database(*args):
    return pd.DataFrame([{'Mouse_ID':'VF101','Date':pd.Timestamp('2026-09-01'),
        'Protocol':1,'Version':version,'Session Dur':600,'Number of Bouts':2,
        'Number of Rewarded Licks':3,'Correct Bouts':[True],'Probas':'90/30'} for version in [1,2,10]])
def load_behavior_catalog(*args): return load_behavior_database(*args)
def load_behavior_mouse(*args): return load_behavior_database(*args)
def render_no_data(): pass
def render_behavior_overview(*args): st.markdown('Mouse overview')
def render_behavior_progression_preview(*args): pass
def behavior_session_info_values(row): return [('Info', 'value')] * 6
def render_behavior_session_info(values): st.session_state['metric_calls'] += len(values)
@contextmanager
def behavior_section_card(*args):
    with st.container(): yield
def quiet_metric(*args): st.session_state['metric_calls'] += 1
def count_valid_bouts(row): return 1
def build_behavior_plot(*args):
    st.session_state['plot_calls'] += 1
    return None
def plot_card(*args, **kwargs): pass
def behavior_plot_card(*args,**kwargs): pass
def load_behavior_video_session(*args): return None
class Prefetch:
    def submit(self,*args): pass
def data_prefetcher(): return Prefetch()
'''
        app = AppTest.from_string(fixtures + callback + '\n' + render + '\nrender_behavior_view("Overview", "VF101")')
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["plot_calls"], 4)
        self.assertEqual(app.session_state["metric_calls"], 0)
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual([tab.label for tab in app.tabs], ["Overview", "Sessions"])
        app.session_state["behavior_tabs"] = "Progression"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["plot_calls"], 8)
        self.assertEqual(app.session_state["behavior_tabs"], "Overview")
        self.assertEqual(app.session_state["metric_calls"], 0)
        app.session_state["behavior_tabs"] = "Sessions"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["plot_calls"], 8)
        self.assertEqual(app.session_state["metric_calls"], 0)
        self.assertIsNone(app.selectbox[0].value)
        self.assertEqual([label.rsplit("v",1)[-1] for label in app.selectbox[0].options], ["10", "2", "1"])

        app.selectbox[0].select_index(2).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["plot_calls"], 11)
        self.assertEqual(app.session_state["metric_calls"], 6)
        chosen = app.selectbox[0].value
        app.session_state["behavior_tabs"] = "Performance"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 0)
        app.session_state["behavior_tabs"] = "Sessions"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, chosen)
