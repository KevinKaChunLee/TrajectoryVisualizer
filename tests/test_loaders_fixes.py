"""Regression tests for the loaders.py fix cluster (B1-B5, R1-R3, C1, C27).

Each test pins a specific defect from the adversarially-verified fix specs so
it cannot silently regress:

- B1: non-object top-level JSON must return {"_error": ...}, never raise.
- B2: dangling function_call (no output before end-of-stream) must surface as
  an error tool part; outputs arriving after task_complete must not be dropped.
- B3: numeric/non-string timestamps degrade to None instead of raising.
- B4: Codex tool status — structured metadata.exit_code is authoritative and
  benign wording ("Found 0 errors") no longer trips the substring heuristic.
- C1: CodeArts legacy-JSON exports are detected as codearts and refused with
  an explicit _error (no parser exists yet) instead of a silent unknown path.
"""

import hashlib
import json
import os
import tempfile
import unittest

from trajviz.insight.loaders import (
    FORMAT_LABELS,
    _iso_to_epoch_ms,
    detect_format,
    load_trajectory,
)
from trajviz.insight.parser import parse_steps


def _codex_event(t, ts, payload):
    return {"type": t, "timestamp": ts, "payload": payload}


def _write(tmp, name, obj, jsonl=False, raw_text=None):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        if raw_text is not None:
            f.write(raw_text)
        elif jsonl:
            for e in obj:
                f.write(json.dumps(e) + "\n")
        else:
            json.dump(obj, f)
    return p


def _codex_session(extra_events):
    """A minimal Codex rollout: session_meta + user prompt + extra_events."""
    return [
        _codex_event("session_meta", "2026-01-05T12:00:00.000Z",
                     {"id": "s1", "cwd": "/p/proj", "cli_version": "1.2.3",
                      "model": "gpt-5", "model_provider": "openai"}),
        _codex_event("response_item", "2026-01-05T12:00:01.000Z",
                     {"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": "fix it"}]}),
    ] + extra_events


def _all_tool_calls(raw):
    return [t for s in parse_steps(raw) for t in s["tool_calls"]]


class B1TopLevelNonObjectTests(unittest.TestCase):
    """B1: load_trajectory honors the _error contract for non-dict JSON."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_top_level_list_returns_error_dict_without_raising(self):
        p = _write(self.tmp, "bare-list.json", [{"role": "user"}])
        result = load_trajectory(p)  # must not raise AttributeError
        self.assertIsInstance(result, dict)
        self.assertIn("_error", result)
        self.assertIn("list", result["_error"])

    def test_top_level_scalar_returns_error_dict(self):
        for name, text, typename in (
            ("num.json", "42", "int"),
            ("str.json", '"hello"', "str"),
        ):
            with self.subTest(name=name):
                p = _write(self.tmp, name, None, raw_text=text)
                result = load_trajectory(p)
                self.assertIn("_error", result)
                self.assertIn(typename, result["_error"])

    def test_detect_format_tolerates_non_dict(self):
        # Defense-in-depth guard for direct detect_format callers.
        self.assertEqual(detect_format([]), "unknown")
        self.assertEqual(detect_format("x"), "unknown")
        self.assertEqual(detect_format(None), "unknown")


class B3NumericTimestampTests(unittest.TestCase):
    """B3: non-string timestamps degrade to None instead of raising."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_numeric_timestamp_returns_none(self):
        self.assertIsNone(_iso_to_epoch_ms(1712345678000))
        self.assertIsNone(_iso_to_epoch_ms(1712345678.5))
        self.assertIsNone(_iso_to_epoch_ms(True))

    def test_valid_iso_string_still_parses(self):
        self.assertEqual(_iso_to_epoch_ms("1970-01-01T00:00:01Z"), 1000)

    def test_falsy_inputs_return_none(self):
        self.assertIsNone(_iso_to_epoch_ms(None))
        self.assertIsNone(_iso_to_epoch_ms(""))
        self.assertIsNone(_iso_to_epoch_ms(0))

    def test_ccsession_with_numeric_timestamp_loads_without_crash(self):
        raw = {
            "format": "ccsession-trajectory",
            "trajectory": [
                {"role": "assistant", "timestamp": 1712345678000,
                 "content": [{"type": "text", "text": "hello"}]},
            ],
        }
        p = _write(self.tmp, "cc-numeric-ts.json", raw)
        loaded = load_trajectory(p)  # must not raise AttributeError
        self.assertNotIn("_error", loaded)
        steps = parse_steps(loaded)
        self.assertEqual(len(steps), 1)
        self.assertIsNone(steps[0]["time_created_ms"])


class B4CodexStatusHeuristicTests(unittest.TestCase):
    """B4: metadata exit_code is authoritative; substring heuristic anchored."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _load_with_output(self, output):
        events = _codex_session([
            _codex_event("response_item", "2026-01-05T12:00:02.000Z",
                         {"type": "function_call", "call_id": "c1",
                          "name": "exec_command",
                          "arguments": json.dumps({"cmd": "npx tsc"})}),
            _codex_event("response_item", "2026-01-05T12:00:03.000Z",
                         {"type": "function_call_output", "call_id": "c1",
                          "output": output}),
            _codex_event("event_msg", "2026-01-05T12:00:04.000Z",
                         {"type": "task_complete"}),
        ])
        return load_trajectory(_write(self.tmp, "rollout.jsonl", events, jsonl=True))

    def _status(self, output):
        calls = _all_tool_calls(self._load_with_output(output))
        self.assertEqual(len(calls), 1)
        return calls[0]["status"]

    def test_benign_error_wording_is_success(self):
        self.assertEqual(
            self._status("All checks passed! Found 0 errors in 12 files."),
            "success")

    def test_error_free_wording_is_success(self):
        self.assertEqual(self._status("Build completed, error-free."), "success")

    def test_metadata_exit_code_zero_is_authoritative_over_error_text(self):
        output = json.dumps({"output": "Error: transient warning noise",
                             "metadata": {"exit_code": 0}})
        self.assertEqual(self._status(output), "success")

    def test_metadata_nonzero_exit_code_is_error_despite_benign_text(self):
        output = json.dumps({"output": "Found 0 errors",
                             "metadata": {"exit_code": 2}})
        self.assertEqual(self._status(output), "error")

    def test_anchored_fallback_still_flags_real_errors(self):
        self.assertEqual(self._status("Error: file not found"), "error")
        self.assertEqual(
            self._status("Traceback (most recent call last):\n  ..."), "error")

    def test_exited_with_code_fallback_preserved(self):
        self.assertEqual(self._status("process exited with code 2"), "error")
        self.assertEqual(self._status("process exited with code 0"), "success")


class B2DanglingToolCallTests(unittest.TestCase):
    """B2: dangling calls surface as error parts; late outputs are kept."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_dangling_function_call_surfaces_as_error_part(self):
        # Session interrupted mid-command: function_call with no output.
        events = _codex_session([
            _codex_event("response_item", "2026-01-05T12:00:02.000Z",
                         {"type": "function_call", "call_id": "c-dangling",
                          "name": "exec_command",
                          "arguments": json.dumps({"cmd": "pytest -x"})}),
        ])
        raw = load_trajectory(_write(self.tmp, "interrupted.jsonl", events, jsonl=True))
        calls = _all_tool_calls(raw)
        self.assertEqual([t["tool_id"] for t in calls], ["c-dangling"])
        self.assertEqual(calls[0]["status"], "error")

    def test_output_after_task_complete_is_not_dropped(self):
        # task_complete flushes the turn (role reset); a straggler output must
        # still be emitted by the final flush, not dropped by the role guard.
        events = _codex_session([
            _codex_event("response_item", "2026-01-05T12:00:02.000Z",
                         {"type": "function_call", "call_id": "c-late",
                          "name": "exec_command",
                          "arguments": json.dumps({"cmd": "make build"})}),
            _codex_event("event_msg", "2026-01-05T12:00:03.000Z",
                         {"type": "task_complete"}),
            _codex_event("response_item", "2026-01-05T12:00:04.000Z",
                         {"type": "function_call_output", "call_id": "c-late",
                          "output": "done"}),
        ])
        raw = load_trajectory(_write(self.tmp, "late-output.jsonl", events, jsonl=True))
        calls = _all_tool_calls(raw)
        self.assertEqual([t["tool_id"] for t in calls], ["c-late"])
        self.assertEqual(calls[0]["status"], "success")
        self.assertEqual(calls[0]["output"], "done")


class C1LegacyCodeArtsTests(unittest.TestCase):
    """C1: legacy CodeArts exports are detected and refused explicitly."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.legacy = {
            "format": "codearts",
            "export_metadata": {"schema_version": 2,
                                "source_format": "codearts_legacy_json"},
            "messages": [{"sender": "user", "content": "hello",
                          "timestamp": "2026-01-05T12:00:00.000Z"}],
        }

    def test_legacy_export_detected_as_codearts(self):
        self.assertEqual(detect_format(self.legacy), "codearts")

    def test_legacy_export_load_returns_explicit_error(self):
        p = _write(self.tmp, "legacy.json", self.legacy)
        result = load_trajectory(p)
        self.assertIn("_error", result)
        self.assertIn("codearts_legacy_json", result["_error"])

    def test_sqlite_export_still_detected(self):
        # Widening the branch must not regress the sqlite-export arm.
        sqlite_export = {
            "export_metadata": {"schema_version": 2,
                                "source_format": "codearts_opencode_sqlite"},
            "info": {"id": "s1"},
            "messages": [],
        }
        self.assertEqual(detect_format(sqlite_export), "codearts")


class SourceShaContractTests(unittest.TestCase):
    """Inviolable: every loaded trajectory carries the sha256 of the exact
    bytes that were parsed (single-read contract)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_json_path_attaches_sha_of_parsed_bytes(self):
        raw = {"format": "ccsession-trajectory", "trajectory": []}
        p = _write(self.tmp, "cc.json", raw)
        with open(p, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        loaded = load_trajectory(p)
        self.assertEqual(loaded["_source_sha256"], expected)
        self.assertEqual(loaded["_source_path"], p)

    def test_codex_jsonl_path_attaches_sha_of_parsed_bytes(self):
        events = _codex_session([
            _codex_event("event_msg", "2026-01-05T12:00:04.000Z",
                         {"type": "task_complete"}),
        ])
        p = _write(self.tmp, "rollout.jsonl", events, jsonl=True)
        with open(p, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        loaded = load_trajectory(p)
        self.assertEqual(loaded["_source_sha256"], expected)


class FormatLabelsTests(unittest.TestCase):
    """C27 (structural part): FORMAT_LABELS is the single source of truth."""

    def test_covers_all_detectable_formats(self):
        self.assertEqual(
            FORMAT_LABELS,
            {"ccsession": "Claude Code", "codearts": "CodeArts",
             "opencode": "OpenCode", "codex": "Codex CLI", "pi": "Pi"},
        )


if __name__ == "__main__":
    unittest.main()
