"""Tests for N-run scorecard + behavioral comparison."""

import json
import os
import tempfile
import unittest

from trajviz.converge.canonical import CanonicalAction
from trajviz.insight.run_group import (
    build_behavioral_comparison,
    build_run_group_scorecard,
    build_run_group_scorecard_html,
    build_run_scorecard_row,
    default_run_label,
    extract_capability_usage,
    normalize_run_paths,
    _parse_mcp_label,
    _parse_skill_name,
)


def _minimal_raw(
    *,
    steps: int = 3,
    tokens_each: int = 100,
    finish: str = "stop",
    read_path: str = "a.py",
    extra_search: bool = False,
) -> dict:
    """Tiny synthetic trajectory for scorecard / behavior tests."""
    messages = []
    t = 1_000
    for i in range(steps):
        role = "user" if i % 2 == 0 else "assistant"
        info = {
            "role": role,
            "id": f"m{i}",
            "sessionID": "ses_test",
            "time": {"created": t, "completed": t + 500},
            "tokens": {
                "total": tokens_each if role == "assistant" else 0,
                "input": tokens_each if role == "assistant" else 0,
                "output": 0,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        }
        if role == "assistant":
            info["finish"] = finish
            info["agent"] = "build"
        parts: list[dict] = [{"type": "text", "text": f"msg {i}"}]
        if role == "assistant" and i == 1:
            parts.append({
                "type": "tool",
                "tool": "read",
                "callID": "c1",
                "state": {
                    "status": "completed",
                    "input": {"filePath": read_path},
                    "output": "ok",
                    "time": {"start": t, "end": t + 100},
                },
            })
            if extra_search:
                parts.append({
                    "type": "tool",
                    "tool": "grep",
                    "callID": "c2",
                    "state": {
                        "status": "completed",
                        "input": {"pattern": "TODO", "path": "src"},
                        "output": "hits",
                        "time": {"start": t, "end": t + 120},
                    },
                })
        messages.append({"info": info, "parts": parts})
        t += 1000
    return {
        "timing": {
            "started_at": "2024-01-01T00:00:00Z",
            "finished_at": "2024-01-01T00:01:00Z",
            "total_duration": 60,
        },
        "messages": messages,
    }


def _action(atype: str, target: str, step: int = 0) -> CanonicalAction:
    return CanonicalAction(step_index=step, action_type=atype, target=target, tool=atype)


def _steps_with_tools(calls: list[tuple[str, dict | None]]) -> list[dict]:
    """Minimal parsed steps carrying tool_calls for capability extraction."""
    tool_calls = []
    for i, (name, inp) in enumerate(calls):
        tool_calls.append({
            "tool_name": name,
            "input": inp or {},
            "status": "completed",
        })
    return [{
        "index": 0,
        "role": "assistant",
        "tool_calls": tool_calls,
        "tokens": {"total": 10, "input": 10, "output": 0, "reasoning": 0,
                   "cache_read": 0, "cache_write": 0},
    }]


class CapabilityExtractionTests(unittest.TestCase):
    def test_mcp_label(self):
        self.assertEqual(
            _parse_mcp_label("mcp__github__list_issues"),
            "github / list_issues",
        )
        self.assertEqual(
            _parse_mcp_label("mcp__plugin_asana_asana__create_task"),
            "plugin_asana_asana / create_task",
        )
        self.assertIsNone(_parse_mcp_label("Read"))

    def test_skill_name(self):
        self.assertEqual(
            _parse_skill_name("Skill", {"skill": "create-rule"}),
            "create-rule",
        )
        self.assertEqual(
            _parse_skill_name("skill", {"name": "canvas"}),
            "canvas",
        )
        self.assertIsNone(_parse_skill_name("Bash", {"command": "ls"}))

    def test_extract_splits_tools_mcps_skills(self):
        steps = _steps_with_tools([
            ("Read", {"file_path": "a.py"}),
            ("mcp__slack__post_message", {"text": "hi"}),
            ("Skill", {"skill": "create-hook"}),
            ("Skill", {"skill": "create-hook"}),
            ("Bash", {"command": "ls"}),
        ])
        usage = extract_capability_usage(steps)
        self.assertEqual(usage["tools"]["Read"], 1)
        self.assertEqual(usage["tools"]["Bash"], 1)
        self.assertEqual(usage["tools"]["Skill"], 2)
        self.assertNotIn("mcp__slack__post_message", usage["tools"])
        self.assertEqual(usage["mcps"]["slack / post_message"], 1)
        self.assertEqual(usage["skills"]["create-hook"], 2)


class NormalizePathsTests(unittest.TestCase):
    def test_list_and_dedupe(self):
        self.assertEqual(normalize_run_paths(["a.json", "b.json", "a.json"]), ["a.json", "b.json"])

    def test_single_and_none(self):
        self.assertEqual(normalize_run_paths("only.json"), ["only.json"])
        self.assertEqual(normalize_run_paths(None), [])

    def test_default_label(self):
        self.assertEqual(default_run_label("/tmp/claude-v2.json"), "claude-v2")


class ScorecardRowTests(unittest.TestCase):
    def test_row_from_raw(self):
        raw = _minimal_raw(steps=4, tokens_each=50)
        row = build_run_scorecard_row(raw, path="/tmp/run_a.json", label="run-a")
        self.assertEqual(row["label"], "run-a")
        self.assertIsNone(row["error"])
        self.assertTrue(row["finished"])
        self.assertEqual(row["steps"], 4)
        self.assertGreaterEqual(row["tokens"], 50)
        self.assertGreaterEqual(row["tool_calls"], 1)

    def test_error_row(self):
        row = build_run_scorecard_row({"_error": "boom"}, path="x.json")
        self.assertEqual(row["error"], "boom")
        self.assertIsNone(row["steps"])


class BehavioralComparisonTests(unittest.TestCase):
    def test_similarity_and_consensus(self):
        runs = [
            {
                "run_id": "a",
                "label": "A",
                "actions": [
                    _action("FILE_READ", "src/main.py", 1),
                    _action("FILE_WRITE", "src/main.py", 2),
                    _action("SEARCH", "bug@src", 3),
                ],
                "steps": _steps_with_tools([
                    ("Read", {}),
                    ("Skill", {"skill": "shared-skill"}),
                    ("mcp__github__list_issues", {}),
                ]),
            },
            {
                "run_id": "b",
                "label": "B",
                "actions": [
                    _action("FILE_READ", "src/main.py", 1),
                    _action("FILE_WRITE", "src/main.py", 2),
                    _action("FILE_READ", "other.py", 3),
                ],
                "steps": _steps_with_tools([
                    ("Read", {}),
                    ("Write", {}),
                    ("Skill", {"skill": "shared-skill"}),
                    ("Skill", {"skill": "only-b"}),
                    ("mcp__slack__post_message", {}),
                ]),
            },
            {
                "run_id": "c",
                "label": "C",
                "actions": [
                    _action("FILE_READ", "src/main.py", 1),
                    _action("FILE_WRITE", "src/main.py", 2),
                ],
                "steps": _steps_with_tools([
                    ("Read", {}),
                    ("Skill", {"skill": "shared-skill"}),
                ]),
            },
        ]
        behavior = build_behavioral_comparison(runs)
        self.assertEqual(behavior["baseline_run_id"], "a")
        self.assertEqual(behavior["similarity"]["a"]["a"], 1.0)
        self.assertGreater(behavior["similarity"]["a"]["c"], 0.5)
        self.assertTrue(
            any(
                r["type"] == "FILE_READ" and r["target"] == "src/main.py"
                for r in behavior["action_matrix"]
            )
        )
        paths = {r["path"]: r for r in behavior["file_matrix"]}
        self.assertIn("src/main.py", paths)
        self.assertEqual(paths["src/main.py"]["kind"], "consensus")
        self.assertTrue(paths["src/main.py"]["cells"]["a"]["read"])
        self.assertTrue(paths["src/main.py"]["cells"]["a"]["write"])
        self.assertIn("other.py", paths)
        self.assertEqual(paths["other.py"]["kind"], "unique")
        self.assertTrue(paths["other.py"]["cells"]["b"]["read"])
        self.assertFalse(paths["other.py"]["cells"]["a"]["read"])
        action_by_key = {
            (r["type"], r["target"]): r for r in behavior["action_matrix"]
        }
        self.assertEqual(action_by_key[("FILE_READ", "other.py")]["kind"], "unique")
        self.assertTrue(action_by_key[("FILE_READ", "other.py")]["cells"]["b"]["present"])
        self.assertFalse(action_by_key[("FILE_READ", "other.py")]["cells"]["a"]["present"])

        tools = {r["key"]: r for r in behavior["tool_matrix"]}
        self.assertIn("Read", tools)
        self.assertEqual(tools["Read"]["kind"], "consensus")
        self.assertEqual(tools["Write"]["kind"], "unique")
        mcps = {r["key"]: r for r in behavior["mcp_matrix"]}
        self.assertIn("github / list_issues", mcps)
        self.assertEqual(mcps["github / list_issues"]["kind"], "unique")
        self.assertIn("slack / post_message", mcps)
        skills = {r["key"]: r for r in behavior["skill_matrix"]}
        self.assertEqual(skills["shared-skill"]["kind"], "consensus")
        self.assertEqual(skills["only-b"]["kind"], "unique")

        # B and C have pattern slots vs baseline A
        self.assertIn("b", behavior["patterns_vs_baseline"])
        self.assertIn("c", behavior["patterns_vs_baseline"])
        html = build_run_group_scorecard_html({
            "rows": [
                {"run_id": "a", "label": "A", "error": None, "finished": True,
                 "steps": 1, "wall_clock_s": 1, "wall_clock_fmt": "1s",
                 "tokens": 1, "tool_calls": 1, "tool_success_pct": 100,
                 "peak_occupancy": 0, "peak_pct": None, "compactions": 0,
                 "format": "test"},
                {"run_id": "b", "label": "B", "error": None, "finished": True,
                 "steps": 2, "wall_clock_s": 2, "wall_clock_fmt": "2s",
                 "tokens": 2, "tool_calls": 2, "tool_success_pct": 100,
                 "peak_occupancy": 0, "peak_pct": None, "compactions": 0,
                 "format": "test"},
            ],
            "behavior": behavior,
            "ok": True,
            "error": None,
        })
        self.assertIn("File coverage", html)
        self.assertIn("Action coverage", html)
        self.assertIn("Tool coverage", html)
        self.assertIn("MCP coverage", html)
        self.assertIn("Skill coverage", html)
        self.assertIn("rg-badge-r", html)
        self.assertIn("rg-badge-w", html)
        self.assertIn("rg-atype-read", html)
        self.assertIn("other.py", html)
        self.assertIn("shared-skill", html)


class ScorecardGroupTests(unittest.TestCase):
    def test_requires_two_paths(self):
        result = build_run_group_scorecard([])
        self.assertFalse(result["ok"])
        html = build_run_group_scorecard_html(result)
        self.assertIn("two", html.lower())

    def test_two_temp_files_include_behavior(self):
        raw_a = _minimal_raw(steps=4, tokens_each=200, read_path="shared.py")
        raw_b = _minimal_raw(
            steps=4, tokens_each=80, read_path="shared.py", extra_search=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "model-a.json")
            path_b = os.path.join(tmp, "model-b.json")
            with open(path_a, "w", encoding="utf-8") as f:
                json.dump(raw_a, f)
            with open(path_b, "w", encoding="utf-8") as f:
                json.dump(raw_b, f)
            result = build_run_group_scorecard(
                [path_a, path_b],
                labels=["Model A", "Model B"],
            )
            self.assertTrue(result["ok"], result.get("error"))
            self.assertIsNotNone(result.get("behavior"))
            behavior = result["behavior"]
            self.assertEqual(behavior["labels"]["model-a"], "Model A")
            self.assertIn("model-b", behavior["similarity"]["model-a"])
            html = build_run_group_scorecard_html(result)
            self.assertIn("Behavioral similarity", html)
            self.assertIn("Model A", html)
            self.assertIn("Consensus", html)

    def test_fixture_plus_copy(self):
        fixture = os.path.join(
            os.path.dirname(__file__), "fixtures", "codearts_minimal.json",
        )
        if not os.path.isfile(fixture):
            self.skipTest("codearts_minimal.json missing")
        with tempfile.TemporaryDirectory() as tmp:
            twin = os.path.join(tmp, "codearts_twin.json")
            with open(fixture, encoding="utf-8") as src, open(twin, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            result = build_run_group_scorecard([fixture, twin])
            self.assertTrue(result["ok"], result.get("error"))
            self.assertIsNotNone(result.get("behavior"))
            # Identical twins → F1 near 1
            ids = result["behavior"]["run_ids"]
            self.assertEqual(len(ids), 2)
            f1 = result["behavior"]["similarity"][ids[0]][ids[1]]
            self.assertGreaterEqual(f1, 0.9)


if __name__ == "__main__":
    unittest.main()
