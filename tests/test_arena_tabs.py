import ast
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class ArenaTabsTests(unittest.TestCase):
    def test_overview_combines_all_tests_and_sessions_remains_separate(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
        render = ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name == 'render_openfield_view'))
        fixtures = '''
import streamlit as st
from dashboard.ui_states import dashboard_tabs
from types import SimpleNamespace
_arena_metrics=SimpleNamespace(collect_metrics=lambda inventory, legacy: (legacy, []))
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import contextmanager
inventory = pd.DataFrame([dict(mouse='VF101', date='2026-10-01', assay='OFT', videos=[], csv_files=[])])
OPENFIELD_ASSAY_CONFIG = {assay:dict(table=assay, color='#123456') for assay in ['OFT','LDT','SIT']}
st.session_state.setdefault('_metric_reads', [])
st.session_state.setdefault('_session_reads', 0)
def openfield_summary(*args): return 'Arena summary'
def render_no_data(): st.info('No data')
@contextmanager
def behavior_section_card(*args):
    with st.container(): yield
def behavior_metrics_table(table):
    st.session_state['_metric_reads'].append(table)
    return pd.DataFrame([dict(mouse='VF101', score=1)])
def openfield_metric_columns(*args): return ['score']
def render_openfield_metric_grid(*args): st.markdown('Metrics')
def openfield_session_metrics(*args):
    st.session_state['_session_reads'] += 1
    return pd.DataFrame([dict(score=1)]), pd.Series(dict(duration_min=5, fps=50, score=1))
def render_behavior_session_info(values):
    st.session_state["_info_cards"]=values
    st.markdown(str(values))
'''
        app = AppTest.from_string(fixtures + render + '\nrender_openfield_view("VF101", inventory)')
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual([tab.label for tab in app.tabs], ['Overview', 'Sessions'])
        self.assertEqual(app.session_state['_metric_reads'], ['OFT', 'LDT', 'SIT'])
        self.assertEqual(app.session_state['_session_reads'], 0)
        app.session_state['arena_tabs'] = 'Sessions'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['_session_reads'], 0)
        self.assertIsNone(app.selectbox[0].value)
        self.assertNotIn('_info_cards', app.session_state)
        app.selectbox[0].select('OFT  |  2026-10-01').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['_session_reads'], 1)
        self.assertIn(('Date', '2026-10-01'), app.session_state['_info_cards'])
        self.assertIn(('Test', 'OFT'), app.session_state['_info_cards'])
        self.assertEqual(app.session_state['_metric_reads'], ['OFT', 'LDT', 'SIT'] * 3)
        # Existing bookmarks/state for the removed panel migrate to Overview.
        app.session_state['arena_tabs'] = 'Global metrics'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['arena_tabs'], 'Overview')
        self.assertEqual(app.session_state['_metric_reads'], ['OFT', 'LDT', 'SIT'] * 4)
