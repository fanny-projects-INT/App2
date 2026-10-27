import ast
import html
import json
import unittest
from pathlib import Path


class ReplayControlsTests(unittest.TestCase):
    def replay_html(self, behavior_only):
        source=(Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                      and node.name=='build_behavior_replay_html')
        namespace={'html':html,'json':json}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'replay','exec'),namespace)
        return namespace['build_behavior_replay_html'](
            {'mouse':'VF101','date':'2026-10-08','duration':600},
            behavior_only=behavior_only,compact=behavior_only)

    def test_behavior_controls_are_under_mouse_and_unique(self):
        markup=self.replay_html(True)
        self.assertEqual(markup.count('id="scrubber"'),1)
        self.assertLess(markup.index('<div class="controls">'),
                        markup.index('<div class="top-item histology-item">'))
        self.assertGreater(markup.index('<div class="controls">'),markup.index('id="scene"'))
        self.assertNotIn('__VIDEO_CONTROLS__',markup)
        self.assertNotIn('__GLOBAL_CONTROLS__',markup)

    def test_ephys_controls_remain_below_all_replay_panels(self):
        markup=self.replay_html(False)
        self.assertEqual(markup.count('id="scrubber"'),1)
        self.assertGreater(markup.index('<div class="controls">'),
                           markup.index('<div class="timeline-stack">'))
