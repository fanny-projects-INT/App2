import ast
import dashboard
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

class InfoImportCompatibilityTests(unittest.TestCase):
    def test_reload_stale_module_on_running_server(self):
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        start = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.ImportFrom)
                     and node.module == "dashboard"
                     and any(a.name == "alyx_info" for a in node.names))
        nodes = tree.body[start:start+4]
        stale = types.ModuleType("dashboard.alyx_info")
        stale.render_info_view = lambda: None
        fresh = types.ModuleType("dashboard.alyx_info")
        fresh.render_info_view = lambda: None
        fresh.prefetch_recent_info = lambda: None
        fresh.INFO_UI_VERSION = 17
        with patch.object(dashboard, "alyx_info", stale), patch("importlib.reload", return_value=fresh) as reload:
            ns = {}
            exec(compile(ast.Module(body=nodes,type_ignores=[]), "app-import", "exec"), ns)
            reload.assert_called_once_with(stale)
            self.assertIs(ns["prefetch_recent_info"], fresh.prefetch_recent_info)
        with patch.object(dashboard, "alyx_info", fresh), patch("importlib.reload") as reload:
            exec(compile(ast.Module(body=nodes,type_ignores=[]), "app-import", "exec"), {})
            reload.assert_not_called()
