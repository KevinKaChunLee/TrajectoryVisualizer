"""Overview Issues triage panel — collect, rank, render."""

import unittest
from types import SimpleNamespace

from trajviz.insight.presenters.issues import (
    OverviewIssue,
    build_overview_issues_html,
    collect_overview_issues,
    rank_issues,
    render_overview_issues_html,
)


def _session(**kwargs):
    base = dict(
        steps=[],
        failure_patterns=[],
        fruitless_streaks=[],
        tool_selection=[],
        plan_metrics={},
        bottleneck_explanations=[],
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class OverviewIssuesTests(unittest.TestCase):
    def test_healthy_empty_state(self):
        html = build_overview_issues_html(_session())
        self.assertIn("overview-issues", html)
        self.assertIn("<details", html)
        self.assertIn("overview-issues-summary", html)
        self.assertIn("No major workflow issues detected", html)
        self.assertNotIn("insight-step-link", html)
        # Empty panel stays collapsed
        self.assertNotIn("<details class='overview-issues-panel' id='overview-issues' open", html)

    def test_issues_panel_starts_expanded(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="x",
                detail="y",
                why="z",
                fix="do the thing",
                steps=(1,),
                source_id="t",
            )
        ])
        self.assertIn("<details class='overview-issues-panel' id='overview-issues' open>", html)
        self.assertIn("1 issue", html)

    def test_errors_rank_above_antipatterns(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "Bash: exit 1",
                    "count": 2,
                    "example_error": "command failed",
                    "recovery_path": ["Read", "Edit"],
                    "steps": [5, 9],
                }],
                fruitless_streaks=[{
                    "start_step": 4,
                    "end_step": 6,
                    "length": 3,
                }],
                bottleneck_explanations=[{
                    "step_idx": 1,
                    "duration": 90.0,
                    "explanation": "Step 1: 90.0s — tool wait",
                }],
            )
        )
        shown = rank_issues(issues)
        self.assertEqual(shown[0].kind, "error")
        self.assertEqual(shown[1].kind, "antipattern")
        self.assertLess(shown[0].severity, shown[1].severity)
        self.assertTrue(all(i.kind != "bottleneck" for i in shown))

    def test_step_chips_and_single_fix_not_a_list(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="Bash: exit 1 (1×)",
                detail="boom",
                why="Typical recovery: Read",
                fix="Encode this recovery in the system prompt: Read → Edit.",
                also="Before Bash: verify paths with Glob/Read.",
                steps=(3, 7),
                source_id="fail:0",
            )
        ])
        self.assertIn("tvGotoWorkflowStep(3)", html)
        self.assertIn("tvGotoWorkflowStep(7)", html)
        self.assertIn("#3", html)
        self.assertIn("#7", html)
        self.assertIn("Encode this recovery", html)
        self.assertIn("Also:", html)
        self.assertNotIn("<ol", html)
        self.assertNotIn("<ul", html)
        self.assertNotIn("<li", html)

    def test_failure_prescription_uses_recovery_path(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "Bash: exit 1",
                    "count": 2,
                    "example_error": "command failed",
                    "recovery_path": ["Read", "Edit"],
                    "steps": [5, 9],
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("Read → Edit", issues[0].fix)
        self.assertTrue(issues[0].also)

    def test_skips_slow_steps_from_issues(self):
        issues = collect_overview_issues(
            _session(
                bottleneck_explanations=[{
                    "step_idx": 12,
                    "duration": 40.0,
                    "explanation": "Step 12: 40.0s — tools",
                    "decomposition": {
                        "tool_pct": 80.0,
                        "inference_pct": 15.0,
                        "idle_pct": 5.0,
                        "dominant_tool": {
                            "name": "Bash",
                            "target": "npm test",
                            "duration_s": 38.0,
                        },
                    },
                }],
            )
        )
        self.assertEqual(issues, [])

    def test_shows_all_ranked_issues(self):
        issues = [
            OverviewIssue(
                kind="antipattern",
                title=f"Waste pattern #{i}",
                detail="x",
                why="y",
                fix="z",
                steps=(i,),
                source_id=f"antipattern:{i}",
            )
            for i in range(12)
        ]
        shown = rank_issues(issues)
        self.assertEqual(len(shown), 12)
        html = render_overview_issues_html(shown)
        self.assertIn("12 issues", html)
        self.assertNotIn("more in Patterns", html)

    def test_skips_tool_error_rollup_when_failure_patterns_exist(self):
        session = _session(
            failure_patterns=[{
                "cluster_label": "Bash: exit 1",
                "count": 1,
                "example_error": "fail",
                "recovery_path": None,
                "steps": [2],
            }],
            steps=[{
                "index": 2,
                "tool_calls": [{"error_type": "platform"}],
            }],
        )
        issues = collect_overview_issues(session)
        kinds_titles = [(i.kind, i.title) for i in issues]
        self.assertEqual(len([t for k, t in kinds_titles if k == "error"]), 1)
        self.assertNotIn("tool error(s)", "".join(t for _, t in kinds_titles))

    def test_fruitless_streak_becomes_antipattern_issue(self):
        issues = collect_overview_issues(
            _session(
                fruitless_streaks=[{
                    "start_step": 4,
                    "end_step": 6,
                    "length": 3,
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "antipattern")
        self.assertIn(4, issues[0].steps)
        self.assertIn(6, issues[0].steps)
        self.assertIn("Glob", issues[0].fix)


if __name__ == "__main__":
    unittest.main()
