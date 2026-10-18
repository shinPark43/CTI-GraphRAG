#!/usr/bin/env python3
"""Turn the design document's HTML export into a compact, print-ready PDF.

The Claude Docs PDF export uses large type and breaks pages around every table,
so this restyles the HTML export (11 pt, tight tables, KaTeX for the formulas,
the architecture diagram as an image) and prints it with headless Chrome.

    uv run python scripts/build_design_pdf.py EXPORT.html ARCH.png reports/ShinhyukPark_GraphRAG_DesignDoc.pdf
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"

CSS = """
@page { size: letter; margin: 0.55in 0.65in; }
body { font: 9.8pt/1.3 "Charter", "Georgia", serif; color: #111; margin: 0; }
body > ul:last-of-type { margin-bottom: 0; }
h1 { font-size: 18pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 11pt 0 3pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 8pt 0 2pt; break-after: avoid; }
p, ul, ol { margin: 0 0 5pt; }
li { margin: 0.5pt 0; }
ul, ol { padding-left: 18pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 8.5pt; line-height: 1.25; }
th, td { border: 0.5pt solid #bbb; padding: 2pt 4pt; vertical-align: top; text-align: left; }
th { background: #f1f1ee; }
tr { break-inside: avoid; }
code { font: 8.6pt "Menlo", monospace; }
pre { background: #f6f6f4; border: 0.5pt solid #ddd; padding: 5pt 7pt; margin: 4pt 0 8pt;
      white-space: pre-wrap; }
pre code { font-size: 7.4pt; line-height: 1.25; }
h3#references + ul { font-size: 8.5pt; line-height: 1.25; }
.math { margin: 4pt 0 8pt; text-align: center; font-size: 10pt; break-inside: avoid; }
pre.mermaid { background: none; border: none; text-align: center; padding: 0; break-inside: avoid; }
pre.mermaid svg { width: 100% !important; max-width: 100% !important; height: auto; max-height: 7.5in; }
figure { margin: 6pt 0 8pt; text-align: center; break-inside: avoid; }
figure img { width: 70%; }
figcaption { font-size: 8.5pt; color: #555; }
a { color: #1a4f8b; text-decoration: none; }
.byline { color: #444; margin-bottom: 10pt; }
"""


def build(export: Path, diagram: Path, out: Path, author: str, date: str) -> None:
    html = export.read_text(encoding="utf-8")
    # byline: date chip and mention chip -> plain author and date
    html = re.sub(r"<p><time[^>]*>.*?</time>.*?</span></p>", f'<p class="byline">{author} · {date}</p>', html,
                  count=1, flags=re.S)
    html = re.sub(r"<figure data-embed=[^>]*><figcaption>(.*?)</figcaption></figure>",
                  r'<figure><img src="arch.png"><figcaption>Figure 1. System architecture: build time and query time.'
                  r'</figcaption></figure>', html, count=1)
    html = re.sub(r'<pre><code class="language-latex">(.*?)</code></pre>', r'<div class="math">\1</div>', html,
                  flags=re.S)
    html = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>', r'<pre class="mermaid">\1</pre>', html,
                  flags=re.S)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<script src="{MERMAID}"></script>
<style>{CSS}</style></head><body>{html}
<script>
document.querySelectorAll('.math').forEach(el => katex.render(el.textContent, el, {{displayMode: true, throwOnError: false}}));
mermaid.initialize({{startOnLoad: false, theme: "neutral", fontFamily: "Helvetica, Arial, sans-serif", themeVariables: {{fontSize: "15px"}}, classDiagram: {{useMaxWidth: true}}}});
mermaid.run({{querySelector: 'pre.mermaid'}});
</script></body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(diagram, Path(tmp) / "arch.png")
        src = Path(tmp) / "doc.html"
        src.write_text(page, encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", f"--print-to-pdf={out.resolve()}", src.as_uri()],
                       check=True, capture_output=True)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path, help="HTML export of the design document")
    ap.add_argument("diagram", type=Path, help="PNG render of the architecture diagram")
    ap.add_argument("out", type=Path)
    ap.add_argument("--author", default="Shinhyuk Park")
    ap.add_argument("--date", default="October 17, 2026")
    args = ap.parse_args()
    build(args.export, args.diagram, args.out, args.author, args.date)


if __name__ == "__main__":
    main()
