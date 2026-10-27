import ast
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class EphysLazyLoadingTests(unittest.TestCase):
    def test_selection_skips_analysis_and_anatomy_skips_atlas_render(self):
        tree = ast.parse((Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8'))
        render = '\n'.join(ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name==name)) for name in ['render_probe_selection_scene', 'render_ephys_view'])
        fixtures = '''
import streamlit as st
import json
from dashboard.ui_states import dashboard_tabs
from dashboard.probe_profile import prepare_profile, build_profile_figure, PROFILE_METRICS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from tempfile import mkdtemp
from contextlib import contextmanager
if 'fixture_root' not in st.session_state: st.session_state['fixture_root']=mkdtemp()
ATLAS_DIR=Path(st.session_state['fixture_root']); (ATLAS_DIR/'meshes').mkdir(exist_ok=True); (ATLAS_DIR/'meshes'/'997.obj').touch()
EPHYS_ROOT=ATLAS_DIR
BEHAVIOR_DB=ATLAS_DIR/'absent.feather'
NAVY='#263548'
inventory=pd.DataFrame([dict(mouse='VF101', date='2026-01-01', probe='probe00', path=str(ATLAS_DIR/'locations.json'), session_dir=str(ATLAS_DIR), source='IBL adjusted')])
def load_probe_tracks(*args): return [dict(date='2026-01-01',probe='probe00',good_units=1)]
def channel_locations_signature(*args): return ()
def resolve_probe_targets(*args): return {}
def atlas_mesh_signature(*args): return ()
def load_obj_mesh(*args): return {}
def build_probe_figure(*args,**kwargs):
    st.session_state['atlas_calls']=st.session_state.get('atlas_calls',0)+1
    return go.Figure()
def _probe_selection_component(**kwargs):
    st.session_state['selection_component_key']=kwargs['key']
    st.session_state['selection_mounts']=st.session_state.get('selection_mounts',0)+1
    return None
def session_color(*args): return '#123456'
def set_ephys_session(key,value): st.session_state[key]=value
def probe_analysis_signature(*args): return ()
def load_probe_analysis(*args, **kwargs):
    st.session_state['analysis_calls']=st.session_state.get('analysis_calls',0)+1
    st.session_state['include_activity']=kwargs.get('include_activity',True)
    return dict(activity_heatmap=None, peri_bout=None, units=pd.DataFrame({'unit_id':[0], 'bombcell_label':['good'], 'depth_um':[20.], 'brain_region':['MOs'], 'firing_rate':[2.], 'amplitude_median':[15.]}),alignment={},channels=pd.DataFrame({'depth_um':[0.,20.], 'brain_region':['MOs','MOs']}))
@contextmanager
def section_card(*args):
    with st.container(): yield
behavior_section_card=section_card
def render_histology_view(*args): st.session_state['histology_mounts']=st.session_state.get('histology_mounts',0)+1
def render_probe_selector(*args): pass
def brain_region_color(*args): return '#123456'
def render_behavior_session_info(*args): pass
def build_qc_figure(*args): return go.Figure()
def build_qc_selectivity_figure(*args): return None
'''
        app=AppTest.from_string(fixtures+'\n'+render+'\nrender_ephys_view("VF101", inventory)')
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['atlas_calls'],1)
        self.assertNotIn('analysis_calls',app.session_state)
        app.session_state['ephys_tabs']='Anatomy'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['atlas_calls'],1)
        self.assertEqual(app.session_state['analysis_calls'],1)
        self.assertEqual(app.session_state['selection_mounts'],2)
        self.assertFalse(app.session_state['include_activity'])
        self.assertEqual(app.session_state['ephys_tabs'], 'Quality control')
        self.assertNotIn('Probe profile', [tab.label for tab in app.tabs])
        self.assertIn('Good unit count', app.selectbox[0].options)
        app.session_state['ephys_tabs']='Probe selection'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['analysis_calls'],1)

        # Each visited tab continues to emit its content while hidden.
        app.session_state['ephys_tabs']='Histology'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['histology_mounts'],1)
        app.session_state['ephys_tabs']='Activity'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['analysis_calls'],2)
        self.assertEqual(app.session_state['histology_mounts'],2)
        self.assertTrue(app.session_state['include_activity'])
        app.session_state['ephys_tabs']='Overview'
        app.run()
        self.assertFalse(app.exception)
        app.session_state['ephys_tabs']='Quality control'
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['analysis_calls'],2)
        self.assertEqual(app.session_state['histology_mounts'],4)
        self.assertEqual(app.session_state['ephys_visited_tabs_VF101']['tabs'],
                         {'Overview', 'Activity', 'Quality control'})
        messages=[info.value for info in app.info]
        self.assertIn('Behavioral database unavailable for this replay.',messages)
        self.assertIn('Good-unit spike clusters are not available for this probe.',messages)
