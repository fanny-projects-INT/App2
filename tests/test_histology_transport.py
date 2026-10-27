import ast
import base64
from io import BytesIO
from pathlib import Path
from unittest import TestCase
from PIL import Image


class HistologyTransportTests(TestCase):
    def test_compact_preview_keeps_geometry_and_inline_image(self):
        node = next(n for n in ast.parse(Path('app.py').read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'compact_histology_preview')
        node.decorator_list = []
        namespace = dict(base64=base64, BytesIO=BytesIO, Image=Image)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'app.py', 'exec'), namespace)
        buffer = BytesIO()
        Image.new('RGB', (16, 8), (100, 50, 0)).save(buffer, format='PNG')
        payload = dict(image=base64.b64encode(buffer.getvalue()).decode(), count=2,
                       width=8, height=8, columns=2, bounds=[[0, 0, 1, 1]] * 2,
                       scales=[[0, 255], [0, 255]], available=[True, True])
        result = namespace['compact_histology_preview'](payload)
        self.assertEqual((result['width'], result['height']), (4, 4))
        self.assertEqual(result['bounds'], payload['bounds'])
        self.assertEqual(result['scales'], payload['scales'])
        with Image.open(BytesIO(base64.b64decode(result['image']))) as image:
            self.assertEqual(image.size, (8, 4))
            self.assertEqual(image.getpixel((0, 0)), (100, 50, 0))
        self.assertEqual(payload['width'], 8)

    def test_browser_uses_inline_images(self):
        source = Path('assets/components/histology/index.html').read_text(encoding='utf-8')
        self.assertIn("image.src='data:image/png;base64,'+payload.image", source)
