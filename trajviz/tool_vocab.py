"""Shared tool-name vocabulary — the single source of truth.

Three modules previously carried hand-maintained copies of "which tools write
files" (insight/patterns.py, insight/metrics.py, converge/canonical.py) plus a
fourth partial check in insight/diagnostics.py — and all four had drifted
pairwise (MultiEdit invisible to converge, NotebookEdit/patch missing from the
edit-precision metric, notebook edits absent from the target-file footprint).
Every consumer now imports from here; add new scaffold tool names HERE only.
"""

# Tools whose successful invocation writes file content. Mixed casings are the
# literal spellings emitted by the supported scaffolds (Claude Code, OpenCode,
# CodeArts, Codex CLI adapters).
WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "Edit", "edit", "Write", "write",
    "NotebookEdit", "patch",
    "MultiEdit", "multiedit",
    "str_replace_editor", "create_file",
})

# Path-bearing input keys used by write tools across scaffolds, in precedence
# order (Claude Code file_path/notebook_path; OpenCode filePath; text-editor
# tools use bare path).
WRITE_PATH_KEYS: tuple[str, ...] = (
    "file_path", "filePath", "notebook_path", "notebookPath", "path",
)


def write_target_path(tool_input: dict) -> str:
    """The file path a write-tool call targets ('' when absent/malformed)."""
    if not isinstance(tool_input, dict):
        return ""
    for key in WRITE_PATH_KEYS:
        value = tool_input.get(key)
        if value:
            return str(value)
    return ""
