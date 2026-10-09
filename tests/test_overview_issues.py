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
        failure_chains=[],
        fruitless_streaks=[],
        tool_selection=[],
        plan_metrics={},
        plan_history=[],
        edit_thrash=[],
        repeated_searches=[],
        phase_regressions=[],
        performance_bottlenecks=[],
        premature_compactions=[],
        file_interactions=[],
        format="",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class OverviewIssuesTests(unittest.TestCase):
    def test_healthy_empty_state(self):
        html = build_overview_issues_html(_session())
        self.assertIn("overview-issues", html)
        self.assertIn("<details", html)
        self.assertIn("overview-issues-summary", html)
        self.assertIn("未检测到明显的工作流问题", html)
        self.assertNotIn("insight-step-link", html)
        self.assertNotIn("<details class='overview-issues-panel' id='overview-issues' open", html)

    def test_issues_panel_starts_expanded(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="x",
                detail="y",
                why="z",
                steps=(1,),
            )
        ])
        self.assertIn("<details class='overview-issues-panel' id='overview-issues' open>", html)
        self.assertIn("1 个问题", html)
        self.assertNotIn("修改位置", html)
        self.assertNotIn("修复建议", html)

    def test_errors_rank_above_antipatterns(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "npm: exit code 1",
                    "count": 2,
                    "example_error": "command failed",
                    "recovery_path": ["Read", "Edit"],
                    "steps": [5, 9],
                    "error_class": "tool",
                }],
                fruitless_streaks=[{
                    "start_step": 4,
                    "end_step": 6,
                    "length": 3,
                }],
                performance_bottlenecks=[{
                    "step_idx": 8,
                    "duration": 42.5,
                    "cause": "tool",
                    "title": "Tool bottleneck: Bash: npm test (40.0s)",
                    "detail": "Step 8: 42.5s — 40s executing tools",
                    "why": "tool wait",
                }],
            )
        )
        shown = rank_issues(issues)
        self.assertEqual(shown[0].kind, "error")
        self.assertEqual(shown[1].kind, "antipattern")
        self.assertEqual(shown[2].kind, "bottleneck")

    def test_system_scaffold_miss_is_low_risk_antipattern(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "Grep: No matches found",
                    "count": 2,
                    "example_error": "No matches found",
                    "recovery_path": None,
                    "steps": [3, 4],
                    "error_class": "system",
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "antipattern")
        self.assertIn("系统错误", issues[0].title)
        self.assertNotIn("频发", issues[0].title)
        self.assertIn("low risk", issues[0].why)

    def test_frequent_system_errors_call_out_volume(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[{
                    "cluster_label": "Read: ENOENT",
                    "count": 6,
                    "example_error": "ENOENT",
                    "recovery_path": None,
                    "steps": [1, 2, 3, 4, 5, 6],
                    "error_class": "system",
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "antipattern")
        self.assertIn("系统错误频发", issues[0].title)
        self.assertIn("6 次", issues[0].title)
        self.assertIn("volume", issues[0].why)

    def test_bottleneck_issue_from_performance_bottlenecks(self):
        issues = collect_overview_issues(
            _session(
                performance_bottlenecks=[{
                    "step_idx": 12,
                    "duration": 18.2,
                    "cause": "tool",
                    "title": "Tool bottleneck: Bash: npm test (14.0s)",
                    "detail": "Step 12: 18.2s — 15s executing tools (Bash: npm test 14s)",
                    "why": "Most of this outlier step was spent in a tool.",
                    "decomposition": {
                        "tool_s": 15.0, "inference_s": 3.2, "idle_s": 0,
                        "timing_incomplete": False,
                        "dominant_tool": {
                            "name": "Bash", "target": "npm test", "duration_s": 14.0,
                        },
                    },
                }],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "bottleneck")
        self.assertEqual(issues[0].steps, (12,))
        self.assertEqual(issues[0].title, "工具瓶颈：Bash: npm test（15.0 秒）")
        self.assertEqual(
            issues[0].detail,
            "第12步耗时 18.2 秒：工具执行 15.0 秒（Bash: npm test 14.0 秒），模型推理 3.2 秒",
        )

    def test_idle_bottleneck_names_the_step_it_precedes(self):
        issues = collect_overview_issues(
            _session(
                performance_bottlenecks=[{
                    "step_idx": 7,
                    "duration": 4.0,
                    "cause": "idle",
                    "decomposition": {
                        "tool_s": 0, "inference_s": 4.0, "idle_s": 30.0,
                        "timing_incomplete": True, "dominant_tool": None,
                    },
                }],
            )
        )
        self.assertEqual(issues[0].title, "空闲/排队瓶颈：第7步前等待 30 秒")
        self.assertTrue(issues[0].detail.startswith("第7步耗时 4.0 秒：开始前空闲 30.0 秒"))
        self.assertTrue(issues[0].detail.endswith("（计时不完整）"))

    def test_synthetic_error_patterns_are_translated_raw_output_is_not(self):
        issues = collect_overview_issues(
            _session(
                failure_patterns=[
                    {
                        "cluster_label": "npm: exit code 1",
                        "count": 2,
                        "example_error": "exit code 1",
                        "steps": [3],
                    },
                    {
                        "cluster_label": "Read: ENOENT: no such file or directory",
                        "count": 1,
                        "example_error": "ENOENT: no such file or directory",
                        "steps": [5],
                    },
                ],
            )
        )
        self.assertEqual(issues[0].title, "npm: 退出码 1（2 次）")
        self.assertEqual(issues[0].detail, "退出码 1")
        self.assertEqual(issues[1].title, "Read: ENOENT: no such file or directory（1 次）")
        self.assertEqual(issues[1].detail, "ENOENT: no such file or directory")

    def test_progress_banner_visible_while_judging(self):
        html = render_overview_issues_html(
            [
                OverviewIssue(
                    kind="error",
                    title="Bash: exit 1 (1×)",
                    detail="boom",
                    why="x",
                    steps=(3,),
                )
            ],
            progress="正在生成修复建议 1/3 — Bash: exit 1 (1×)",
        )
        self.assertIn("overview-issues-progress", html)
        self.assertIn("思考中…", html)
        self.assertIn("overview-issues-progress-spinner", html)
        self.assertIn("animateTransform", html)
        self.assertIn("currentColor", html)
        self.assertIn("正在生成修复建议 1/3", html)
        self.assertIn("1 个问题 · 正在生成修复建议…", html)
        self.assertNotIn("修改位置", html)

    def test_step_chips_without_fix_or_change(self):
        html = render_overview_issues_html([
            OverviewIssue(
                kind="error",
                title="Bash: exit 1 (1×)",
                detail="boom",
                why="Typical recovery: Read",
                steps=(3, 7),
            )
        ])
        self.assertIn("tvGotoWorkflowStep(3)", html)
        self.assertIn("tvGotoWorkflowStep(7)", html)
        self.assertIn(">第3步<", html)
        self.assertIn(">第7步<", html)
        self.assertNotIn(">#3<", html)
        self.assertNotIn("Why it matters", html)
        self.assertNotIn("Typical recovery", html)
        self.assertNotIn("修改位置", html)
        self.assertNotIn("修复建议", html)
        self.assertNotIn("补充：", html)

    def test_step_chips_overflow_reads_in_chinese(self):
        html = render_overview_issues_html([
            OverviewIssue(kind="error", title="x", detail="", steps=tuple(range(10))),
        ])
        self.assertIn(">第7步<", html)
        self.assertNotIn(">第8步<", html)
        self.assertIn("另有 2 步", html)
        self.assertNotIn("more", html)

    def test_cards_are_numbered_in_ranked_order(self):
        ranked = rank_issues([
            OverviewIssue(kind="bottleneck", title="slow", detail="", steps=(1,)),
            OverviewIssue(kind="error", title="broken", detail="", steps=(9,)),
            OverviewIssue(kind="antipattern", title="wasteful", detail="", steps=(2,)),
        ])
        html = render_overview_issues_html(ranked)
        positions = [
            html.find(f">问题 {n}</span><span class='overview-issue-title'>{title}<")
            for n, title in ((1, "broken"), (2, "wasteful"), (3, "slow"))
        ]
        self.assertNotIn(-1, positions)
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("问题 4", html)

    def test_failure_shows_recovery_as_why_not_fix(self):
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
        self.assertIn("Read → Edit", issues[0].why)
        html = render_overview_issues_html(issues)
        self.assertNotIn("Read → Edit", html)
        self.assertNotIn("修复建议", html)
        self.assertNotIn("修改位置", html)

    def test_shows_all_ranked_issues(self):
        issues = [
            OverviewIssue(
                kind="antipattern",
                title=f"Waste pattern #{i}",
                detail="x",
                why="y",
                steps=(i,),
            )
            for i in range(12)
        ]
        shown = rank_issues(issues)
        self.assertEqual(len(shown), 12)
        html = render_overview_issues_html(shown)
        self.assertIn("12 个问题", html)
        self.assertIn("问题 12", html)
        self.assertNotIn("Show all", html)
        self.assertNotIn("overview-issues-remainder", html)
        for i in range(12):
            self.assertIn(f"Waste pattern #{i}", html)

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
        self.assertNotIn("工具调用错误", "".join(t for _, t in kinds_titles))

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

    def test_premature_compactions_become_antipattern_issues(self):
        issues = collect_overview_issues(
            _session(
                premature_compactions=[
                    {"step": 139, "occupancy_before": 11_941, "occupancy_after": 12_766,
                     "before_pct": 9.3, "window_limit": 128_000, "grew": True,
                     "reason": "window_grew"},
                    {"step": 245, "occupancy_before": 41_935, "occupancy_after": 12_130,
                     "before_pct": 32.8, "window_limit": 128_000, "grew": False,
                     "reason": "low_occupancy"},
                    {"step": 259, "occupancy_before": 29_140, "occupancy_after": 12_162,
                     "before_pct": 22.8, "window_limit": 128_000, "grew": False,
                     "reason": "low_occupancy"},
                ],
            )
        )
        titles = [i.title for i in issues]
        self.assertTrue(any("压缩后上下文反而变大" in t for t in titles))
        self.assertTrue(any("过早压缩" in t for t in titles))
        grew = next(i for i in issues if "压缩后上下文反而变大" in i.title)
        premature = next(i for i in issues if "过早压缩" in i.title)
        self.assertEqual(grew.steps, (139,))
        self.assertIn("第139步", grew.detail)
        self.assertEqual(premature.steps, (245, 259))
        self.assertIn("23–33%", premature.detail)
        self.assertIn("128k", premature.detail)

    def test_failure_cascade_skips_single_step_chains(self):
        issues = collect_overview_issues(
            _session(
                failure_chains=[
                    {"start": 1, "end": 1, "steps": [1]},
                    {"start": 3, "end": 5, "steps": [3, 4, 5]},
                ],
            )
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].kind, "error")
        self.assertEqual(issues[0].title, "连续失败（3 步）")
        self.assertEqual(issues[0].detail, "第3–5步")
        self.assertEqual(issues[0].steps, (3, 4, 5))

    def test_stalled_plan_items_are_called_todo_items(self):
        issues = collect_overview_issues(
            _session(
                plan_metrics={"stalled": [
                    {"content": "Write tests", "start_step": 3, "end_step": 4},
                ]},
            )
        )
        self.assertEqual(issues[0].title, "停滞的待办项（1 项）")
        self.assertEqual(issues[0].detail, "“Write tests”")

    def test_plan_resets_edit_thrash_repeated_search(self):
        issues = collect_overview_issues(
            _session(
                plan_metrics={"plan_resets": 2, "stalled": []},
                plan_history=[{"step": 2, "items": []}, {"step": 10, "items": []}],
                edit_thrash=[{
                    "path": "src/app.py",
                    "count": 4,
                    "fail_count": 2,
                    "steps": [5, 6, 7, 8],
                    "start_step": 5,
                    "end_step": 8,
                }],
                repeated_searches=[{
                    "signature": "grep:foo|",
                    "display": "foo|",
                    "count": 3,
                    "steps": [1, 4, 9],
                }],
            )
        )
        titles = [i.title for i in issues]
        self.assertTrue(any("计划重置" in t for t in titles))
        self.assertTrue(any("编辑失败后反复重试" in t for t in titles))
        self.assertTrue(any("重复的空搜索" in t for t in titles))
        thrash = next(i for i in issues if "编辑失败后反复重试" in i.title)
        self.assertEqual(thrash.steps, (5, 6, 7, 8))
        self.assertEqual(thrash.detail, "第5–8步中有 2 次写入失败")
        for title in titles:
            self.assertFalse(
                title[:1].isdigit(),
                f"issue title should not start with a digit: {title!r}",
            )


if __name__ == "__main__":
    unittest.main()
