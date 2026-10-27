import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


class ArenaVideoOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        selected = [
            node for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "OPENFIELD_SKELETON_CORE_POINTS"
                        for target in node.targets)
            ) or (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in {
                    "load_openfield_skeleton_overlay",
                    "build_openfield_video_overlay_html",
                }
            )
        ]
        cache_data = lambda *args, **kwargs: (lambda function: function)
        namespace = {
            "np": np,
            "pd": pd,
            "json": __import__("json"),
            "st": SimpleNamespace(cache_data=cache_data),
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), "app.py", "exec"), namespace)
        cls.load_overlay = staticmethod(namespace["load_openfield_skeleton_overlay"])
        cls.build_html = staticmethod(namespace["build_openfield_video_overlay_html"])

    def test_tracking_is_sampled_and_low_confidence_points_are_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "remapped_cropped_test_clean.csv"
            columns = pd.MultiIndex.from_product(
                [["tracker"], ["nose", "neck"], ["x", "y", "likelihood"]],
                names=["scorer", "bodyparts", "coords"],
            )
            data = np.tile([100.0, 200.0, 1.0], (8, 2))
            data[2, 2] = .1
            pd.DataFrame(data, columns=columns).to_csv(path)
            stats = path.stat()
            payload = self.load_overlay(str(path), stats.st_size, stats.st_mtime_ns, 40)

        self.assertEqual(payload["frames"], [0, 2, 4, 6])
        self.assertEqual(payload["fps"], 40)
        self.assertEqual(payload["corePoints"], ["nose", "neck", "torso", "tail_base"])
        self.assertIsNone(payload["points"]["nose"][1])
        self.assertEqual(payload["points"]["neck"][0], [100.0, 200.0])

    def test_overlay_uses_video_clock_and_does_not_embed_a_second_video(self):
        payload = {"fps": 40, "frames": [0], "points": {}, "corePoints": []}
        markup = self.build_html(payload, "VF093-OFT-2026-08-17")
        self.assertIn("video.currentTime", markup)
        self.assertIn("data-arena-skeleton", markup)
        self.assertNotIn("<video", markup)
        self.assertNotIn("lineTo", markup)
        self.assertIn("video.preload='auto'", markup)
        self.assertNotIn("responseType='blob'", markup)


if __name__ == "__main__":
    unittest.main()
