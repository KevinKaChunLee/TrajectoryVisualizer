"""Standalone HTML report of Overview + charts + Patterns."""

import json
import os
import tempfile
import unittest

from trajviz.insight.loaders import load_trajectory
from trajviz.insight.report import (
    ReportError,
    _mixed_md_to_html,
    build_report_html,
    write_report_file,
)


def _write(tmp, name, obj, jsonl=False):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        if jsonl:
            for e in obj:
                f.write(json.dumps(e) + "\n")
        else:
            json.dump(obj, f)
    return p


class MixedMarkdownTests(unittest.TestCase):
    def test_headers_chips_and_tables(self):
        md = (
            "### Timing\n\n"
            "<div>chip</div>\n\n"
            "**Top latency steps**\n\n"
            "| Step | Role | Duration (s) |\n"
            "|---:|---|---:|\n"
            "| 3 | `assistant` | 1.50 |\n"
            "\n*No data*\n"
        )
        html = _mixed_md_to_html(md)
        self.assertIn("<h3>Timing</h3>", html)
        self.assertIn("<div>chip</div>", html)
        self.assertIn("<strong>Top latency steps</strong>", html)
        self.assertIn("<table>", html)
        self.assertIn("<td><code>assistant</code></td>", html)
        self.assertIn("<em>No data</em>", html)


class ReportBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.oc = {
            "info": {
                "id": "ses_test",
                "directory": "/home/user/proj",
                "time": {"created": 1_000, "updated": 4_000},
            },
            "messages": [
                {
                    "info": {"role": "user", "time": {"created": 1000}},
                    "parts": [{"type": "text", "text": "list the repo"}],
                },
                {
                    "info": {
                        "role": "assistant",
                        "time": {"created": 2000, "completed": 3500},
                        "tokens": {
                            "total": 40,
                            "input": 20,
                            "output": 20,
                            "reasoning": 0,
                            "cache": {"read": 0, "write": 0},
                        },
                    },
                    "parts": [
                        {"type": "text", "text": "running ls"},
                        {
                            "type": "tool",
                            "tool": "bash",
                            "state": "completed",
                            "input": {"command": "ls"},
                            "output": "README.md\n",
                        },
                    ],
                },
            ],
        }

    def _loaded(self):
        return load_trajectory(_write(self.tmp, "oc.json", self.oc))

    def test_empty_raw_raises(self):
        with self.assertRaises(ReportError):
            build_report_html({})

    def test_error_raw_raises(self):
        with self.assertRaises(ReportError):
            build_report_html({"_error": "nope"})

    def test_html_contains_overview_and_patterns_not_workflow_shell(self):
        doc = build_report_html(self._loaded())
        self.assertIn("<!DOCTYPE html>", doc)
        self.assertIn("TrajViz HTML report", doc)
        self.assertIn("OpenCode", doc)
        self.assertIn("oc.json", doc)
        self.assertIn("id='summary'", doc)
        self.assertIn("id='patterns'", doc)
        self.assertIn("Recurring tool sequences", doc)
        self.assertNotIn("wf-filter-chips", doc)
        self.assertNotIn("Load Trajectory", doc)

    def test_plotly_included_once_via_cdn_when_charts_exist(self):
        doc = build_report_html(self._loaded())
        self.assertIn("plotly", doc.lower())
        self.assertEqual(doc.count("cdn.plot.ly"), 1)

    def test_write_report_file_names_from_source(self):
        raw = self._loaded()
        path = write_report_file(raw, self.tmp)
        self.assertTrue(path.endswith("oc-trajviz-report.html"))
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as f:
            self.assertIn("TrajViz HTML report", f.read())

    def test_cli_flag_writes_output(self):
        import sys

        from trajviz.insight.__main__ import main

        src = _write(self.tmp, "cli.json", self.oc)
        dest = os.path.join(self.tmp, "out.html")
        old = sys.argv
        try:
            sys.argv = ["trajviz", "--report", src, "-o", dest]
            main()
        finally:
            sys.argv = old
        self.assertTrue(os.path.isfile(dest))
        with open(dest, encoding="utf-8") as f:
            self.assertIn("cli.json", f.read())

    def test_export_is_not_a_page_load_value_fn(self):
        """DownloadButton(value=fn) runs on Blocks load and toasted an error."""
        import inspect

        from trajviz.insight.insight import build_ui
        from trajviz.insight.ui.upload import prepare_html_export

        src = inspect.getsource(build_ui)
        self.assertNotIn("value=prepare_html_export", src)
        self.assertNotIn("value=_prepare_html_export", src)
        self.assertNotIn("value=_export_html_report", src)

        app = build_ui()
        fns = getattr(app, "fns", None) or app.default_config.fns
        export_fns = [bf for bf in fns.values() if bf.fn is prepare_html_export]
        self.assertTrue(export_fns, "Export HTML prepare handler is missing")
        for bf in export_fns:
            triggers = [name for _, name in bf.targets]
            self.assertNotIn("load", triggers)

    def test_prepare_export_empty_disables_button(self):
        from trajviz.insight.ui.upload import prepare_html_export

        upd = prepare_html_export({}, [], False)
        self.assertFalse(upd.get("interactive", True))
        self.assertIsNone(upd.get("value"))

    def test_prepare_export_writes_file(self):
        from trajviz.insight.ui.upload import prepare_html_export

        raw = self._loaded()
        upd = prepare_html_export(raw, None, False)
        self.assertTrue(upd.get("interactive"))
        path = upd.get("value")
        self.assertTrue(path and path.endswith(".html"))
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as f:
            self.assertIn("TrajViz HTML report", f.read())


if __name__ == "__main__":
    unittest.main()
