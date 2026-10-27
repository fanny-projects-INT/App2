import ast
import unittest
from pathlib import Path


class NativeRenderingTests(unittest.TestCase):
    def test_every_tabbed_view_uses_native_fragments(self):
        root = Path(__file__).resolve().parents[1]
        for file, names in [('app.py', ['render_behavior_view', 'render_ephys_view', 'render_openfield_view']),
                            ('dashboard/alyx_info.py', ['_render_info_tabs'])]:
            tree = ast.parse((root / file).read_text(encoding='utf-8'))
            for name in names:
                view = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                self.assertEqual([ast.unparse(d) for d in view.decorator_list], ['st.fragment'], name)

    def test_shared_ui_has_no_loading_or_dom_controller(self):
        source = (Path(__file__).resolve().parents[1] / 'dashboard' / 'ui_states.py').read_text(encoding='utf-8')
        functions = {n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(functions, {'data_prefetcher', 'render_no_data', 'dashboard_tabs'})
        self.assertNotIn('<script>', source)
        self.assertNotIn('cloneNode', source)

    def test_main_renders_only_the_selected_native_page(self):
        source = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n, ast.Try) and 'render_home_view()' in ast.unparse(n))
        self.assertEqual(main.finalbody, [])
        containers = [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'st.container']
        self.assertEqual(len(containers), 1)
        self.assertIn('page_', ast.unparse(containers[0]))
        self.assertNotIn('unsafe_allow_javascript', ast.unparse(main))
        self.assertNotIn('install_render_gate', source)
        self.assertNotIn('page_payload', source)
