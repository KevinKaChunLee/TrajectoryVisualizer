"""LLM Issues fix judge — pack, parse, render (mocked chat)."""

import unittest
from types import SimpleNamespace

from trajviz.insight.issue_judge import (
    judge_issue_fix,
    judge_overview_issues,
    pack_issue_judge_context,
    parse_issue_judgment,
)
from trajviz.insight.llm_config import AnalysisLLMConfig
from trajviz.insight.presenters.issues import (
    IssueJudgment,
    OverviewIssue,
    collect_overview_issues,
    rank_issues,
    render_overview_issues_html,
)


def _cfg() -> AnalysisLLMConfig:
    return AnalysisLLMConfig(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        provider="openai",
        temperature=0.0,
        max_tokens=512,
        timeout=30,
        source="analyze",
    )


def _session(**kwargs):
    base = dict(
        steps=[],
        failure_patterns=[],
        failure_chains=[],
        fruitless_streaks=[],
        tool_selection=[],
        plan_metrics={},
        plan_history=[],
        edit_thrash=[],
        repeated_searches=[],
        phase_regressions=[],
        bottleneck_explanations=[],
        file_interactions=[],
        format="claude_code",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class ParseJudgmentTests(unittest.TestCase):
    def test_raw_json(self):
        j = parse_issue_judgment(
            '{"where":"CLAUDE.md","fix":"Stop Bash reads","also":"","confidence":"high"}'
        )
        self.assertEqual(j.where, "CLAUDE.md")
        self.assertEqual(j.fix, "Stop Bash reads")
        self.assertEqual(j.confidence, "high")

    def test_fenced_json(self):
        j = parse_issue_judgment(
            """Here you go:
```json
{"where": "Environment", "fix": "Fix permissions", "also": "not a prompt", "confidence": "medium"}
```
"""
        )
        self.assertEqual(j.where, "Environment")
        self.assertIn("permissions", j.fix)

    def test_rejects_missing_fix(self):
        with self.assertRaises(ValueError):
            parse_issue_judgment('{"where":"x","fix":"","also":"","confidence":"low"}')


class PackContextTests(unittest.TestCase):
    def test_includes_workflow_window_and_skills(self):
        session = _session(
            steps=[
                {
                    "index": 4,
                    "role": "assistant",
                    "text_preview": "searching",
                    "tool_calls": [{
                        "tool_name": "Bash",
                        "input": {"command": "cat foo.py"},
                        "error_type": "exit 1",
                    }],
                },
                {
                    "index": 5,
                    "role": "assistant",
                    "text_preview": "retry",
                    "tool_calls": [{"tool_name": "Read", "input": {"file_path": "foo.py"}}],
                },
            ],
            file_interactions=[{
                "step": 3,
                "tool": "Skill",
                "path": "skill:search",
                "type": "skill",
            }],
        )
        issue = OverviewIssue(
            kind="antipattern",
            title="1 Bash-for-reading",
            detail="sed/cat",
            why="shell reads",
            steps=(4,),
            source_id="antipattern:bash_read",
        )
        packed = pack_issue_judge_context(session, issue)
        self.assertIn("Bash-for-reading", packed)
        self.assertIn('"index": 4', packed)
        self.assertIn("cat foo.py", packed)
        self.assertIn("skill:search", packed)
        self.assertIn("claude_code", packed)


class JudgeMockTests(unittest.TestCase):
    def test_judge_issue_fix_uses_chat_fn(self):
        session = _session(steps=[{"index": 1, "role": "assistant", "tool_calls": []}])
        issue = OverviewIssue(
            kind="error",
            title="Bash: exit 1 (1×)",
            detail="fail",
            why="no recovery",
            steps=(1,),
            source_id="fail:0",
        )

        def chat_fn(config, system, messages):
            self.assertIn("fix judge", system.lower())
            self.assertIn("Simplified Chinese", system)
            self.assertIn("Simplified Chinese", messages[0]["content"])
            self.assertEqual(messages[0]["role"], "user")
            return (
                '{"where":"CLAUDE.md","fix":"Bash 读文件前先确认路径",'
                '"also":"","confidence":"high"}'
            )

        j = judge_issue_fix(session, issue, config=_cfg(), chat_fn=chat_fn)
        self.assertEqual(j.where, "CLAUDE.md")
        self.assertIn("Bash", j.fix)

    def test_judge_overview_attaches_judgment_and_html(self):
        session = _session(
            fruitless_streaks=[{
                "start_step": 4,
                "end_step": 6,
                "length": 3,
            }],
            steps=[
                {"index": i, "role": "assistant", "tool_calls": []}
                for i in range(4, 7)
            ],
        )
        issues = rank_issues(collect_overview_issues(session))
        self.assertEqual(len(issues), 1)

        def chat_fn(config, system, messages):
            return (
                '{"where":"CLAUDE.md","fix":"空搜索超过 N 次后停止",'
                '"also":"预先给出路径","confidence":"medium"}'
            )

        judged, errors = judge_overview_issues(
            session, issues, config=_cfg(), chat_fn=chat_fn,
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(judged[0].judgment)
        html = render_overview_issues_html(judged)
        self.assertIn(">Change<", html)
        self.assertIn(">Fix", html)
        self.assertIn("CLAUDE.md", html)
        self.assertIn("空搜索超过 N 次后停止", html)
        self.assertIn("with LLM fix", html)

    def test_render_without_judgment_has_no_change_fix(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="x",
                detail="y",
                why="z",
                steps=(1,),
            )
        ])
        self.assertNotIn(">Change<", html)
        self.assertNotIn(">Fix", html)


if __name__ == "__main__":
    unittest.main()
