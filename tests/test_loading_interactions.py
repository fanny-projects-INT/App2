import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

class LoadingInteractionTests(unittest.TestCase):
    def test_info_tabs_and_arrows_are_in_a_fragment(self):
        source=(Path(__file__).resolve().parents[1]/'dashboard'/'alyx_info.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        tabs=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_render_info_tabs')
        self.assertIn('st.fragment',[ast.unparse(n) for n in tabs.decorator_list])
        self.assertIn('_render_session_list',ast.unparse(tabs))

    def test_all_pages_use_one_tab_navigation_policy(self):
        root=Path(__file__).resolve().parents[1]
        groups={'app.py': {'render_behavior_view':1, 'render_ephys_view':1, 'render_openfield_view':2},
                'dashboard/alyx_info.py': {'_render_info_tabs':1}}
        for file, views in groups.items():
            tree=ast.parse((root/file).read_text(encoding='utf-8'))
            for name, count in views.items():
                view=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                calls=[n for n in ast.walk(view) if isinstance(n,ast.Call) and ast.unparse(n.func)=='dashboard_tabs']
                self.assertEqual(len(calls),count,name)
                self.assertFalse(any(k.arg=='on_change' for call in calls for k in call.keywords),name)
        ui=ast.parse((root/'dashboard'/'ui_states.py').read_text(encoding='utf-8'))
        helper=next(n for n in ui.body if isinstance(n,ast.FunctionDef) and n.name=='dashboard_tabs')
        st=SimpleNamespace(tabs=lambda *args,**kwargs:kwargs,session_state={})
        ns={'st':st}
        exec(compile(ast.Module(body=[helper],type_ignores=[]),'tabs','exec'),ns)
        self.assertEqual(ns['dashboard_tabs'](['One','Two'],key='view'),
                         {'key':'view','on_change':'rerun'})

    def test_hidden_nested_tab_selection_is_restored(self):
        source=(Path(__file__).resolve().parents[1]/'dashboard'/'ui_states.py').read_text(encoding='utf-8')
        helper=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='dashboard_tabs')
        st=SimpleNamespace(tabs=lambda *args,**kwargs:None,session_state={'metrics':'LDT'})
        ns={'st':st}
        exec(compile(ast.Module(body=[helper],type_ignores=[]),'tabs','exec'),ns)
        ns['dashboard_tabs'](['OFT','LDT','SIT'],key='metrics')
        del st.session_state['metrics']  # Streamlit cleans up a widget in a hidden panel.
        ns['dashboard_tabs'](['OFT','LDT','SIT'],key='metrics')
        self.assertEqual(st.session_state['metrics'],'LDT')
