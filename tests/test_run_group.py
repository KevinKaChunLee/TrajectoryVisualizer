"""Tests for N-run scorecard + behavioral comparison."""

import json
import os
import tempfile
import unittest

from trajviz.converge.canonical import CanonicalAction
from trajviz.insight.charts import build_run_group_agent_timeline
from trajviz.insight.run_group import (
    build_behavioral_comparison,
    build_run_group_scorecard,
    build_run_group_scorecard_html,
    build_run_group_behavior_html,
    build_run_scorecard_row,
    default_run_label,
    extract_capability_usage,
    normalize_run_paths,
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

    def test_extract_tools_and_skills(self):
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
        self.assertEqual(usage["tools"]["mcp__slack__post_message"], 1)
        self.assertEqual(usage["skills"]["create-hook"], 2)
        self.assertNotIn("mcps", usage)


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
        self.assertEqual(tools["mcp__github__list_issues"]["kind"], "unique")
        self.assertEqual(tools["mcp__slack__post_message"]["kind"], "unique")
        self.assertNotIn("mcp_matrix", behavior)
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
        self.assertNotIn("MCP coverage", html)
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
        self.assertTrue(
            "two" in html.lower() or "overview" in html.lower(),
            html,
        )

    def test_overview_baseline_plus_one_upload(self):
        baseline = _minimal_raw(steps=4, tokens_each=200, read_path="shared.py")
        baseline["_source_path"] = "/tmp/overview-run.json"
        other = _minimal_raw(
            steps=4, tokens_each=80, read_path="shared.py", extra_search=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path_b = os.path.join(tmp, "alt-model.json")
            with open(path_b, "w", encoding="utf-8") as f:
                json.dump(other, f)
            # One upload is enough when Overview baseline is present
            result = build_run_group_scorecard(
                [path_b],
                baseline_raw=baseline,
            )
            self.assertTrue(result["ok"], result.get("error"))
            self.assertEqual(len(result["rows"]), 2)
            self.assertIn("baseline", result["rows"][0]["label"].lower())
            self.assertEqual(
                result["behavior"]["baseline_run_id"],
                result["rows"][0]["run_id"],
            )
            # Without baseline, one path is not enough
            alone = build_run_group_scorecard([path_b])
            self.assertFalse(alone["ok"])

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
            self.assertEqual(len(result.get("timeline_runs") or []), 2)
            behavior = result["behavior"]
            self.assertEqual(behavior["labels"]["model-a"], "Model A")
            self.assertIn("model-b", behavior["similarity"]["model-a"])
            html = build_run_group_scorecard_html(result, include_behavior=False)
            self.assertIn("Model A", html)
            self.assertNotIn("Behavioral similarity", html)
            behavior_html = build_run_group_behavior_html(result)
            self.assertIn("Behavioral similarity", behavior_html)
            self.assertIn("Consensus", behavior_html)
            fig = build_run_group_agent_timeline(result["timeline_runs"])
            self.assertGreater(len(fig.data), 0)
            self.assertIn("Agent timeline", fig.layout.title.text)

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


class AgentTimelineChartTests(unittest.TestCase):
    def test_one_lane_per_run_colored_by_agent(self):
        runs = [
            {
                "run_id": "a",
                "label": "Run A",
                "steps": [
                    {"index": 0, "role": "user", "agent": "",
                     "tokens": {"total": 0}, "tool_call_count": 0},
                    {"index": 1, "role": "assistant", "agent": "",
                     "tokens": {"total": 10}, "tool_call_count": 1},
                    {"index": 2, "role": "assistant", "agent": "explore",
                     "tokens": {"total": 20}, "tool_call_count": 2},
                    {"index": 3, "role": "assistant", "agent": "explore",
                     "tokens": {"total": 5}, "tool_call_count": 0},
                ],
            },
            {
                "run_id": "b",
                "label": "Run B",
                "steps": [
                    {"index": 0, "role": "assistant", "agent": "",
                     "tokens": {"total": 8}, "tool_call_count": 1},
                    {"index": 1, "role": "assistant", "agent": "",
                     "tokens": {"total": 8}, "tool_call_count": 0},
                ],
            },
        ]
        fig = build_run_group_agent_timeline(runs)
        # At least one bar segment per run; legend includes main
        self.assertGreaterEqual(len(fig.data), 2)
        y_vals = {t.y[0] for t in fig.data if t.y}
        self.assertEqual(y_vals, {"Run A", "Run B"})
        legend_names = {t.name for t in fig.data if t.showlegend}
        self.assertIn("main", legend_names)
        self.assertTrue(any(n.startswith("sub ") for n in legend_names))

    def test_empty_runs(self):
        fig = build_run_group_agent_timeline([])
        self.assertEqual(len(fig.data), 0)


if __name__ == "__main__":
    unittest.main()
