"""DeepSeek Harness (DSH) JSONL loader."""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from trajviz.insight.charts import bind_timeline_agents, build_file_interaction_chart
from trajviz.insight.diagnostics import extract_file_interactions
from trajviz.insight.loaders import _zip_dsh_members, detect_format, load_trajectory
from trajviz.insight.metrics import compute_agent_summary, extract_agent_info
from trajviz.insight.parser import parse_steps


REAL_SAMPLE = Path(
    "/home/user/Downloads/dsh-session-session-90768b91-4a4e-44fb-a995-eff77f7cbfb5"
)


def _write(tmp: str, name: str, events: list) -> str:
    path = os.path.join(tmp, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return path


def _session(**extra):
    event = {
        "type": "session",
        "version": 0,
        "id": "session-parent",
        "createdAt": 1_787_623_647_203,
        "cwd": "/home/user/proj",
        "delegationDepth": 0,
        "agentPreset": "standard",
    }
    event.update(extra)
    return event


def _evt(etype, seq, time, data=None, **extra):
    event = {"type": etype, "seq": seq, "time": time, "data": data or {}}
    event.update(extra)
    return event


def _user(seq, time, text, *, msg_id="u1", kind="user"):
    return _evt("user/message", seq, time, {
        "content": [{"type": "text", "text": text}],
        "source": {"kind": kind},
        "role": "user",
        "id": msg_id,
    })


def _assistant(seq, time, content, *, usage=None, msg_id="a1",
               model="deepseek-v4-pro", provider="deepseek-official"):
    return _evt("assistant/message", seq, time, {
        "turn": 1,
        "step": 1,
        "message": {
            "role": "assistant",
            "content": content,
            "source": {"kind": "model", "provider": provider, "model": model},
            "id": msg_id,
        },
        "usage": usage or {
            "inputTokens": 100,
            "outputTokens": 20,
            "cacheReadTokens": 10,
            "reasoningTokens": 5,
        },
    })


def _tool_call(seq, time, call_id, name, arguments):
    return _evt("tool/call", seq, time, {
        "turn": 1, "step": 1, "callId": call_id, "name": name,
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
    })


def _tool_result(seq, time, call_id, text, *, is_error=False):
    return _evt("tool/result", seq, time, {
        "turn": 1,
        "step": 1,
        "message": {
            "source": {"kind": "tool", "callId": call_id},
            "content": [{
                "type": "tool-result",
                "toolCallId": call_id,
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            }],
            "role": "user",
            "id": f"res-{call_id}",
        },
        **({"error": {"name": "ToolError", "code": "ERR"}} if is_error else {}),
    })


class DshLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.events = [
            _session(),
            _evt("permission/preset", 0, 1_787_623_647_210, {"preset": "workspace-write"}),
            _evt("request/context", 1, 1_787_623_647_211, {
                "provider": "deepseek-official", "model": "deepseek-v4-pro",
            }),
            _evt("session/title", 2, 1_787_623_647_212, {"title": "Summarize the repo"}),
            _user(3, 1_787_623_647_300, "summarize this repo"),
            _user(4, 1_787_623_647_301,
                  "Current runtime context. This snapshot supersedes earlier ones.",
                  msg_id="plugin-1", kind="plugin"),
            _assistant(5, 1_787_623_647_400, [
                {"type": "reasoning", "text": "I should list the directory."},
                {"type": "text", "text": "I'll look around."},
                {"type": "tool-call", "id": "call-1", "name": "bash",
                 "arguments": json.dumps({"command": "ls /home/user/proj"})},
            ]),
            _tool_call(6, 1_787_623_647_401, "call-1", "bash",
                       {"command": "ls /home/user/proj"}),
            _tool_result(7, 1_787_623_647_500, "call-1", "trajviz\nREADME.md"),
            _assistant(8, 1_787_623_647_600, [
                {"type": "tool-call", "id": "call-2", "name": "read",
                 "arguments": json.dumps({"file_path": "/home/user/proj/README.md"})},
            ], usage={"inputTokens": 200, "outputTokens": 40, "cacheReadTokens": 0,
                      "reasoningTokens": 0}, msg_id="a2"),
            _tool_call(9, 1_787_623_647_601, "call-2", "read",
                       {"file_path": "/home/user/proj/README.md"}),
            _tool_result(10, 1_787_623_647_700, "call-2", "# TrajViz"),
            _assistant(11, 1_787_623_647_800, [
                {"type": "tool-call", "id": "call-3", "name": "write",
                 "arguments": json.dumps({
                     "file_path": "/home/user/proj/NOTES.md", "content": "hi",
                 })},
            ], usage={"inputTokens": 210, "outputTokens": 30, "cacheReadTokens": 0,
                      "reasoningTokens": 0}, msg_id="a3"),
            _tool_call(12, 1_787_623_647_801, "call-3", "write",
                       {"file_path": "/home/user/proj/NOTES.md", "content": "hi"}),
            _tool_result(13, 1_787_623_647_900, "call-3", "written"),
            _assistant(14, 1_787_623_648_000, [
                {"type": "tool-call", "id": "call-fork", "name": "subagent_fork",
                 "arguments": json.dumps({
                     "description": "Explore insight", "prompt": "read insight/",
                 })},
            ], usage={"inputTokens": 300, "outputTokens": 15, "cacheReadTokens": 0,
                      "reasoningTokens": 0}, msg_id="a4"),
            _tool_call(15, 1_787_623_648_001, "call-fork", "subagent_fork",
                       {"description": "Explore insight", "prompt": "read insight/"}),
            _tool_result(16, 1_787_623_648_100, "call-fork",
                         "started subagent child-1"),
            _assistant(17, 1_787_623_648_200, [
                {"type": "text", "text": "Here is an overview of the repo."},
            ], usage={"inputTokens": 400, "outputTokens": 80, "cacheReadTokens": 20,
                      "reasoningTokens": 12}, msg_id="a5"),
        ]

    def _load(self, events=None, name="session.jsonl"):
        return load_trajectory(_write(self.tmp, name, events or self.events))

    def test_detects_as_dsh_not_pi(self):
        raw = self._load()
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "dsh")
        self.assertTrue(raw.get("_dsh_format"))
        self.assertFalse(raw.get("_pi_format"))

    def test_does_not_steal_pi_jsonl(self):
        events = [
            {"type": "session", "version": 3, "id": "sess-1",
             "timestamp": "2026-08-24T01:29:43.221Z", "cwd": "/p"},
            {"type": "message", "id": "m1", "timestamp": "2026-08-24T01:30:33.026Z",
             "message": {"role": "user", "content": "hello", "timestamp": 1}},
        ]
        raw = self._load(events, name="pi.jsonl")
        self.assertEqual(detect_format(raw), "pi")

    def test_does_not_steal_codex_jsonl(self):
        events = [
            {"type": "session_meta", "timestamp": "2026-01-05T12:00:00.000Z",
             "payload": {"id": "s1", "cwd": "/p", "model": "gpt-5"}},
        ]
        raw = self._load(events, name="codex.jsonl")
        self.assertEqual(detect_format(raw), "codex")

    def test_session_metadata(self):
        raw = self._load()
        self.assertEqual(raw["metadata"]["session_id"], "session-parent")
        self.assertEqual(raw["metadata"]["directory"], "/home/user/proj")
        self.assertEqual(raw["metadata"]["directory_name"], "proj")
        self.assertEqual(raw["metadata"]["agent"], "dsh")
        self.assertEqual(raw["metadata"]["model"], "deepseek-v4-pro")
        self.assertEqual(raw["info"]["title"], "Summarize the repo")
        self.assertEqual(raw["info"]["version"], "0")
        self.assertEqual(raw["metadata"]["server_version"], "0")

    def test_plugin_user_messages_skipped(self):
        steps = parse_steps(self._load())
        user_previews = [s["text_preview"] for s in steps if s["role"] == "user"]
        self.assertEqual(user_previews, ["summarize this repo"])
        self.assertTrue(all("runtime context" not in (p or "") for p in user_previews))

    def test_thinking_becomes_reasoning(self):
        steps = parse_steps(self._load())
        reasoned = [s for s in steps if s.get("has_reasoning")]
        self.assertTrue(reasoned)
        self.assertTrue(any(
            any(p.get("type") == "reasoning" for p in s["parts"])
            for s in reasoned
        ))

    def test_tool_call_paired_with_result(self):
        steps = parse_steps(self._load())
        bash = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-1"
        )
        self.assertEqual(bash["tool_name"], "Bash")
        self.assertEqual(bash["status"], "success")
        self.assertIn("command", bash["input"])
        self.assertIn("trajviz", bash["output"])

    def test_read_write_normalized_to_file_path(self):
        steps = parse_steps(self._load())
        read = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-2"
        )
        write = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-3"
        )
        self.assertEqual(read["tool_name"], "Read")
        self.assertEqual(read["input"]["file_path"], "/home/user/proj/README.md")
        self.assertEqual(write["tool_name"], "Write")
        self.assertEqual(write["input"]["file_path"], "/home/user/proj/NOTES.md")

    def test_subagent_fork_records_child_session_id(self):
        steps = parse_steps(self._load())
        fork = next(
            tc for s in steps for tc in s["tool_calls"] if tc["tool_id"] == "call-fork"
        )
        self.assertEqual(fork["tool_name"], "Agent")
        self.assertEqual(fork["status"], "success")
        self.assertEqual(fork["metadata"].get("sessionId"), "child-1")

    def test_token_usage_ingested(self):
        steps = parse_steps(self._load())
        first = next(
            s for s in steps
            if s["role"] == "assistant" and s["tokens"]["input"] == 100
        )
        self.assertEqual(first["tokens"]["output"], 20)
        self.assertEqual(first["tokens"]["reasoning"], 5)
        self.assertEqual(first["tokens"]["cache_read"], 10)
        self.assertEqual(first["tokens"]["total"], 100 + 20 + 5 + 10)

    def test_session_header_uses_model(self):
        steps = parse_steps(self._load())
        model_id, provider_id, _ = extract_agent_info(steps)
        self.assertEqual(model_id, "deepseek-v4-pro")
        self.assertEqual(provider_id, "deepseek-official")

    def test_timestamps_are_epoch_ms(self):
        steps = parse_steps(self._load())
        stamped = [s for s in steps if s.get("time_created_ms")]
        self.assertTrue(stamped)
        for step in stamped:
            self.assertIsInstance(step["time_created_ms"], int)
            self.assertGreater(step["time_created_ms"], 10**12)

    def test_dangling_tool_call_marked_error(self):
        events = [
            _session(),
            _assistant(1, 1_787_623_647_400, [
                {"type": "tool-call", "id": "call-x", "name": "bash",
                 "arguments": json.dumps({"command": "pwd"})},
            ]),
        ]
        steps = parse_steps(self._load(events, name="dangling.jsonl"))
        tool = next(tc for s in steps for tc in s["tool_calls"])
        self.assertEqual(tool["status"], "error")
        self.assertEqual(tool["tool_id"], "call-x")

    def test_failed_tool_result_status(self):
        events = [
            _session(),
            _assistant(1, 1_787_623_647_400, [
                {"type": "tool-call", "id": "call-f", "name": "bash",
                 "arguments": json.dumps({"command": "false"})},
            ]),
            _tool_call(2, 1_787_623_647_401, "call-f", "bash", {"command": "false"}),
            _tool_result(3, 1_787_623_647_500, "call-f", "sandbox refused", is_error=True),
        ]
        steps = parse_steps(self._load(events, name="fail.jsonl"))
        tool = next(tc for s in steps for tc in s["tool_calls"])
        self.assertEqual(tool["status"], "error")
        self.assertIn("sandbox refused", tool["output"])

    def test_truncated_final_line_does_not_reject_file(self):
        path = os.path.join(self.tmp, "trunc.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for event in self.events[:5]:
                handle.write(json.dumps(event) + "\n")
            handle.write('{"type":"assistant/message","seq":99,"time":1787')
        raw = load_trajectory(path)
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "dsh")

    def test_directory_resolves_session_jsonl(self):
        session_dir = os.path.join(self.tmp, "dsh-session-session-parent")
        _write(session_dir, "session.jsonl", self.events)
        raw = load_trajectory(session_dir)
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "dsh")

    def test_sibling_subagents_merged_and_seed_dropped(self):
        session_dir = os.path.join(self.tmp, "export")
        _write(session_dir, "session.jsonl", self.events)
        child_events = [
            _session(id="child-1", origin="subagent", parentSession="session-parent",
                     seedLength=5, createdAt=1_787_623_648_050),
            # Inherited parent prefix (seq < seedLength) must be dropped.
            _user(3, 1_787_623_647_300, "summarize this repo", msg_id="seed-user"),
            _assistant(4, 1_787_623_647_400, [
                {"type": "text", "text": "I'll look around."},
            ], msg_id="seed-asst"),
            _user(6, 1_787_623_648_200, "Explore trajviz/insight/", msg_id="child-user"),
            _assistant(7, 1_787_623_648_400, [
                {"type": "text", "text": "Insight package map."},
            ], usage={"inputTokens": 50, "outputTokens": 10, "cacheReadTokens": 0,
                      "reasoningTokens": 0}, msg_id="child-asst"),
        ]
        _write(session_dir, os.path.join("subagents", "child-1", "session.jsonl"),
               child_events)
        raw = load_trajectory(os.path.join(session_dir, "session.jsonl"))
        self.assertEqual(raw["metadata"]["sub_agent_count"], 1)
        steps = parse_steps(raw)
        child_steps = [s for s in steps if s.get("session_id") == "child-1"]
        self.assertTrue(child_steps)
        self.assertTrue(all(s.get("is_sub_agent") for s in child_steps))
        self.assertTrue(all(s.get("parent_session_id") == "session-parent"
                            for s in child_steps))
        child_previews = [s["text_preview"] for s in child_steps]
        self.assertTrue(any("Insight package map" in (p or "") for p in child_previews))
        self.assertFalse(any((p or "") == "I'll look around." for p in child_previews))

    def test_file_interaction_chart_includes_subagent_files(self):
        session_dir = os.path.join(self.tmp, "files")
        _write(session_dir, "session.jsonl", self.events)
        child_events = [
            _session(id="child-1", origin="subagent", parentSession="session-parent",
                     seedLength=1, createdAt=1_787_623_648_050),
            _assistant(2, 1_787_623_648_400, [
                {"type": "tool-call", "id": "c-read", "name": "read",
                 "arguments": json.dumps({"file_path": "/home/user/proj/insight.py"})},
            ], msg_id="ca"),
            _tool_call(3, 1_787_623_648_401, "c-read", "read",
                       {"file_path": "/home/user/proj/insight.py"}),
            _tool_result(4, 1_787_623_648_500, "c-read", "ok"),
        ]
        _write(session_dir, os.path.join("subagents", "child-1", "session.jsonl"),
               child_events)
        steps = parse_steps(load_trajectory(os.path.join(session_dir, "session.jsonl")))
        interactions = extract_file_interactions(steps)
        self.assertTrue(any(i["path"].endswith("insight.py") for i in interactions))
        fig = build_file_interaction_chart(interactions, steps=steps)
        names = {t.name for t in fig.data}
        self.assertIn("main", names)
        self.assertTrue(any(str(n).startswith("sub ") for n in names))
        child_trace = next(t for t in fig.data if str(t.name).startswith("sub "))
        self.assertTrue(any("insight.py" in str(y) for y in (child_trace.y or [])))
        _, labels, _ = bind_timeline_agents(steps)
        self.assertIn("main", {v for k, v in labels.items() if k})
        self.assertFalse(any("standard" in v for v in labels.values()))

    def test_zip_export_merges_subagents(self):
        session_dir = os.path.join(self.tmp, "boxed")
        _write(session_dir, "session.jsonl", self.events)
        child_events = [
            _session(id="child-1", origin="subagent", parentSession="session-parent",
                     seedLength=1, createdAt=1_787_623_648_050),
            _user(2, 1_787_623_648_200, "Explore insight", msg_id="cu"),
            _assistant(3, 1_787_623_648_400, [
                {"type": "text", "text": "child done"},
            ], usage={"inputTokens": 11, "outputTokens": 2, "cacheReadTokens": 0,
                      "reasoningTokens": 0}, msg_id="ca"),
        ]
        _write(session_dir, os.path.join("subagents", "child-1", "session.jsonl"),
               child_events)
        zip_path = os.path.join(self.tmp, "dsh-session-session-parent.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(os.path.join(session_dir, "session.jsonl"),
                          "dsh-session-session-parent/session.jsonl")
            archive.write(
                os.path.join(session_dir, "subagents", "child-1", "session.jsonl"),
                "dsh-session-session-parent/subagents/child-1/session.jsonl",
            )
        raw = load_trajectory(zip_path)
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "dsh")
        self.assertEqual(raw["metadata"]["sub_agent_count"], 1)
        steps = parse_steps(raw)
        self.assertTrue(any(s.get("session_id") == "child-1" for s in steps))

    def _child_events(self, seed_length=1):
        return [
            _session(id="child-1", origin="subagent", parentSession="session-parent",
                     seedLength=seed_length, createdAt=1_787_623_648_050),
            _user(2, 1_787_623_648_200, "Explore insight", msg_id="cu"),
            _assistant(3, 1_787_623_648_400, [
                {"type": "text", "text": "child done"},
            ], usage={"inputTokens": 11, "outputTokens": 2, "cacheReadTokens": 0,
                      "reasoningTokens": 0}, msg_id="ca"),
        ]

    def test_zip_members_gui_root_layout(self):
        parent, children = _zip_dsh_members([
            "session.jsonl",
            "subagents/child-1/session.jsonl",
            "subagents/child-2/session.jsonl",
            "notes.txt",
        ])
        self.assertEqual(parent, "session.jsonl")
        self.assertEqual([cid for cid, _ in children], ["child-1", "child-2"])

    def test_zip_members_ignores_mysubagents_false_positive(self):
        parent, children = _zip_dsh_members(["mysubagents/session.jsonl"])
        self.assertEqual(parent, "mysubagents/session.jsonl")
        self.assertEqual(children, [])

    def test_zip_gui_root_layout_merges_subagents(self):
        """DSH GUI zip: session.jsonl + subagents/<id>/ at archive root."""
        session_dir = os.path.join(self.tmp, "gui-export")
        _write(session_dir, "session.jsonl", self.events)
        _write(session_dir, os.path.join("subagents", "child-1", "session.jsonl"),
               self._child_events())
        zip_path = os.path.join(self.tmp, "gui-export.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(os.path.join(session_dir, "session.jsonl"), "session.jsonl")
            archive.write(
                os.path.join(session_dir, "subagents", "child-1", "session.jsonl"),
                "subagents/child-1/session.jsonl",
            )
        raw = load_trajectory(zip_path)
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "dsh")
        self.assertEqual(raw["metadata"]["sub_agent_count"], 1)
        steps = parse_steps(raw)
        self.assertTrue(any(s.get("session_id") == "child-1" for s in steps))
        summaries = compute_agent_summary(steps, raw)
        labels = {row["label"] for row in summaries}
        self.assertIn("main", labels)
        self.assertTrue(any(row["agent_id"] == "child-1" for row in summaries))

    def test_zip_bytes_without_zip_extension(self):
        session_dir = os.path.join(self.tmp, "noext")
        _write(session_dir, "session.jsonl", self.events)
        _write(session_dir, os.path.join("subagents", "child-1", "session.jsonl"),
               self._child_events())
        zip_path = os.path.join(self.tmp, "noext.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(os.path.join(session_dir, "session.jsonl"), "session.jsonl")
            archive.write(
                os.path.join(session_dir, "subagents", "child-1", "session.jsonl"),
                "subagents/child-1/session.jsonl",
            )
        blob = os.path.join(self.tmp, "gradio-upload")
        with open(zip_path, "rb") as src, open(blob, "wb") as dest:
            dest.write(src.read())
        raw = load_trajectory(blob)
        self.assertNotIn("_error", raw)
        self.assertEqual(raw["metadata"]["sub_agent_count"], 1)

    def test_isolated_jsonl_finds_named_export_dir(self):
        export_root = os.path.join(self.tmp, "downloads")
        session_dir = os.path.join(export_root, "dsh-session-session-parent")
        _write(session_dir, "session.jsonl", self.events)
        _write(session_dir, os.path.join("subagents", "child-1", "session.jsonl"),
               self._child_events())
        isolated = os.path.join(self.tmp, "gradio-cache", "session.jsonl")
        isolated_dir = os.path.dirname(isolated)
        os.makedirs(isolated_dir, exist_ok=True)
        with open(isolated, "w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event) + "\n")
        with patch.dict(os.environ, {"TRAJVIZ_DSH_EXPORT_ROOT": export_root}):
            raw = load_trajectory(isolated)
        self.assertEqual(raw["metadata"]["sub_agent_count"], 1)
        steps = parse_steps(raw)
        self.assertTrue(any(s.get("session_id") == "child-1" for s in steps))

    def test_child_origin_does_not_search_export_dir(self):
        export_root = os.path.join(self.tmp, "decoy-root")
        decoy = os.path.join(export_root, "dsh-session-child-1")
        _write(decoy, os.path.join("subagents", "other", "session.jsonl"), [
            _session(id="other", origin="subagent", parentSession="session-parent",
                     seedLength=1, createdAt=1_787_623_648_050),
            _assistant(2, 1_787_623_648_400, [
                {"type": "text", "text": "should not merge"},
            ], msg_id="decoy"),
        ])
        child_path = _write(
            os.path.join(self.tmp, "orphan-child"), "session.jsonl",
            self._child_events(),
        )
        with patch.dict(os.environ, {"TRAJVIZ_DSH_EXPORT_ROOT": export_root}):
            raw = load_trajectory(child_path)
        self.assertEqual(raw["metadata"]["sub_agent_count"], 0)
        steps = parse_steps(raw)
        self.assertFalse(any("should not merge" in (s.get("text_preview") or "")
                             for s in steps))

    def test_already_converted_object_reloads(self):
        first = self._load()
        dumped = {k: v for k, v in first.items() if not k.startswith("_source")}
        path = os.path.join(self.tmp, "converted.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dumped, handle)
        second = load_trajectory(path)
        self.assertNotIn("_error", second)
        self.assertEqual(detect_format(second), "dsh")
        self.assertTrue(second.get("_dsh_format"))

    def test_format_hint_mismatch_with_pi(self):
        raw = load_trajectory(
            _write(self.tmp, "dsh.jsonl", self.events), format_hint="pi",
        )
        self.assertEqual(raw.get("_error_code"), "mismatch")
        self.assertEqual(raw.get("_detected"), "dsh")

    @unittest.skipUnless(REAL_SAMPLE.is_dir(), "sample DSH export not present")
    def test_real_export_loads(self):
        raw = load_trajectory(str(REAL_SAMPLE))
        self.assertNotIn("_error", raw)
        self.assertEqual(detect_format(raw), "dsh")
        steps = parse_steps(raw)
        self.assertTrue(any(s["role"] == "user" for s in steps))
        self.assertTrue(any(s["role"] == "assistant" for s in steps))
        self.assertGreater(raw["metadata"].get("sub_agent_count", 0), 0)
        self.assertTrue(any(s.get("is_sub_agent") for s in steps))
        tools = {tc["tool_name"] for s in steps for tc in s["tool_calls"]}
        self.assertIn("Read", tools)
        self.assertIn("Agent", tools)


if __name__ == "__main__":
    unittest.main()
