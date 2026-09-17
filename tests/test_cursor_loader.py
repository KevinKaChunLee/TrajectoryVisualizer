from __future__ import annotations

import json
import unittest
from pathlib import Path

from trajviz.insight.loaders import detect_format, load_trajectory
from trajviz.insight.parser import parse_steps
from trajviz.insight.presenters import trajectory_format_label


FIXTURE = Path(__file__).parent / "fixtures" / "cursor_minimal.json"
PARENT_ID = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
CHILD_ID = "cccccccc-4444-5555-6666-dddddddddddd"


class CursorLoaderTests(unittest.TestCase):
    def test_format_is_detected_before_generic_opencode(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(detect_format(raw), "cursor")

        generic = {"info": {"id": "ses_open"}, "messages": []}
        self.assertEqual(detect_format(generic), "opencode")

    def test_loader_preserves_identity_snapshot_and_does_not_invent_tokens(self) -> None:
        loaded = load_trajectory(str(FIXTURE))
        self.assertEqual(detect_format(loaded), "cursor")
        self.assertEqual(loaded["metadata"]["agent"], "cursor")
        self.assertEqual(loaded["metadata"]["generator_name"], "cursor_consolidator")
        self.assertEqual(loaded["metadata"]["session_count"], 2)
        self.assertEqual(loaded["metadata"]["sub_agent_count"], 1)
        self.assertEqual(loaded["metadata"]["token_semantics"], "context_window_snapshot")
        self.assertEqual(loaded["metadata"]["context_snapshot"]["total_used_tokens"], 1200)
        self.assertEqual(loaded["token_usage"], {})
        self.assertFalse(loaded["_capabilities"]["has_runtime_token_usage"])
        self.assertTrue(loaded["_cursor_format"])

    def test_parser_maps_tools_and_subagent_hierarchy(self) -> None:
        steps = parse_steps(load_trajectory(str(FIXTURE)))
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[1]["tool_calls"][0]["tool_name"], "Read")
        self.assertIn("file_path", steps[1]["tool_calls"][0]["input"])
        self.assertEqual(
            steps[1]["tool_calls"][0]["input"]["file_path"],
            "/workspace/demo/app.py",
        )
        self.assertEqual(steps[1]["tool_calls"][1]["tool_name"], "Task")
        self.assertEqual(steps[1]["tool_calls"][1]["metadata"]["sessionId"], CHILD_ID)
        self.assertEqual(steps[1]["session_id"], PARENT_ID)
        self.assertTrue(steps[3]["is_sub_agent"])
        self.assertEqual(steps[3]["parent_session_id"], PARENT_ID)
        self.assertEqual(steps[3]["tool_calls"][0]["tool_name"], "Grep")
        self.assertEqual(steps[1]["duration"], 1.5)

    def test_strreplace_and_shell_map_to_shared_vocab(self) -> None:
        from trajviz.insight.formats.cursor import _convert_cursor_to_internal

        raw = {
            "info": {"id": "c1", "time": {"created": 1, "updated": 2}},
            "messages": [
                {
                    "info": {"role": "assistant", "id": "m1", "sessionID": "c1"},
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "StrReplace",
                            "callID": "e1",
                            "state": {
                                "status": "unknown",
                                "input": {"path": "/tmp/a.py", "old_string": "a", "new_string": "b"},
                                "output": "",
                                "metadata": {},
                            },
                        },
                        {
                            "type": "tool",
                            "tool": "Shell",
                            "callID": "s1",
                            "state": {
                                "status": "unknown",
                                "input": {"command": "ls"},
                                "output": "",
                                "metadata": {},
                            },
                        },
                    ],
                }
            ],
            "export_metadata": {
                "schema_version": 1,
                "source_format": "cursor_composer",
            },
        }
        converted = _convert_cursor_to_internal(raw)
        tools = [part["tool"] for part in converted["messages"][0]["parts"]]
        self.assertEqual(tools, ["Edit", "Bash"])
        self.assertEqual(
            converted["messages"][0]["parts"][0]["state"]["input"]["file_path"],
            "/tmp/a.py",
        )

    def test_human_readable_format_label(self) -> None:
        self.assertEqual(trajectory_format_label("cursor"), "Cursor")
