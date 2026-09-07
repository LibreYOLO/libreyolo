"""Regression checks for the inline-SVG trust boundary; standard library only."""
import tempfile
import unittest
from pathlib import Path
from wrap_svg import wrap


class StaticSvgTests(unittest.TestCase):
    def rejected(self, content):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'model.svg', Path(tmp)/'index.html'
            src.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'+content+'</svg>')
            with self.assertRaises(ValueError):
                wrap(src, dst)
            self.assertFalse(dst.exists())

    def test_active_elements_and_attributes(self):
        for payload in [
            '<script>alert(1)</script>', '<g onload="alert(1)"/>',
            '<foreignObject><div xmlns="http://www.w3.org/1999/xhtml">HTML</div></foreignObject>',
            '<animate attributeName="href" values="javascript:alert(1)"/>',
            '<set attributeName="onload" to="alert(1)"/>',
            '<style>body {display:none}</style>', '<rect style="fill:red"/>',
            '<a href="javascript:alert(1)"><text>click</text></a>',
            '<use href="https://example.com/a.svg#x"/>',
            '<rect fill="url(https://example.com/a.svg#paint)"/>',
            '<image href="https://example.com/a.png"/>',
            '<image href="data:image/svg+xml;base64,PHN2Zy8+"/>',
            '<image href="data:image/png;base64,PHN2Zy8+"/>',
        ]:
            with self.subTest(payload=payload): self.rejected(payload)

    def test_approved_artwork_stays_supported(self):
        artwork = Path(__file__).resolve().parent.parent/'assets/approved-yolo9-t.svg'
        with tempfile.TemporaryDirectory() as tmp:
            dst = wrap(artwork, Path(tmp)/'index.html')
            text = dst.read_text()
            self.assertIn('RepNCSPELAN', text)
            self.assertIn('data:image/png;base64,', text)
            self.assertIn('marker-end="url(#arrow)"', text)

    def test_text_remains_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'model.svg', Path(tmp)/'index.html'
            src.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><title>&lt;script&gt;title&lt;/script&gt;</title><text>&lt;script&gt;label&lt;/script&gt;</text></svg>')
            wrap(src, dst)
            self.assertNotIn('<script>label', dst.read_text())
            self.assertIn('&lt;script&gt;label', dst.read_text())


if __name__ == '__main__':
    unittest.main()
