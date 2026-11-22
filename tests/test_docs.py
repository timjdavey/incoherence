"""Run the code in the README and the docs, and check what it prints.

Every ``python`` block on a page is run in order, in one namespace for that
page. When a ``text`` block comes straight after, it must match what the code
printed.
"""

import contextlib
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(r"^```python\n(.*?)^```\n(?:\n```text\n(.*?)^```\n)?", re.DOTALL | re.MULTILINE)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_code_runs_and_prints_what_the_page_shows(page):
    namespace: dict[str, object] = {}
    for code, expected in BLOCK.findall(page.read_text(encoding="utf-8")):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(code, namespace)
        if expected:
            assert printed.getvalue() == expected
