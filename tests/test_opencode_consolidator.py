from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import opencode_consolidator as consolidator


SCHEMA = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    slug TEXT,
    directory TEXT,
    title TEXT,
    version TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    time_created INTEGER,
    time_updated INTEGER,
    time_compacting INTEGER,
    time_archived INTEGER,
    summary_diffs TEXT,
    revert TEXT,
    permission TEXT
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""


class OpenCodeConsolidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _make_database(self) -> tuple[Path, list[dict]]:
        database = self.root / "opencode.db"
        connection = sqlite3.connect(database)
        connection.executescript(SCHEMA)
        connection.execute(
            """
            INSERT INTO session VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "ses_root",
                "project",
                "workspace",
                None,
                "root",
                "/workspace",
                "Root",
                "test",
                0,
                0,
                0,
                1_000,
                2_000,
                None,
                None,
                None,
                None,
                None,
            ),
        )
        token_objects = [
            {
                "total": 100,
                "input": 20,
                "output": 10,
                "reasoning": 0,
                "cache": {"read": 70, "write": 0},
            },
            {
                "total": 80,
                "input": 10,
                "output": 5,
                "reasoning": 0,
                "cache": {"read": 65, "write": 0},
            },
        ]
        for index, tokens in enumerate(token_objects):
            created = 1_100 + index * 100
            payload = {
                "role": "assistant",
                "agent": "build",
                "tokens": tokens,
                "time": {"created": created, "completed": created + 50},
            }
            connection.execute(
                "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                (
                    f"msg_{index}",
                    "ses_root",
                    created,
                    created + 50,
                    json.dumps(payload),
                ),
            )
        connection.commit()
        connection.close()
        return database, token_objects

    def _run_main(self, database: Path, output: Path) -> None:
        with (
            patch.dict(os.environ, {"OPENCODE_DATABASE": str(database)}),
            patch.object(
                sys,
                "argv",
                ["opencode_consolidator.py", "ses_root", str(output)],
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            consolidator.main()

    def test_preserves_per_message_token_objects(self) -> None:
        database, expected_tokens = self._make_database()
        output = self.root / "trajectory.json"

        self._run_main(database, output)

        result = json.loads(output.read_text(encoding="utf-8"))
        actual_tokens = [message["info"]["tokens"] for message in result["messages"]]
        self.assertEqual(actual_tokens, expected_tokens)

    def test_refuses_to_overwrite_source_database(self) -> None:
        database, _ = self._make_database()
        original_header = database.read_bytes()[:16]

        with self.assertRaises(SystemExit) as raised:
            self._run_main(database, database)

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(database.read_bytes()[:16], original_header)
        check = sqlite3.connect(database)
        try:
            self.assertEqual(check.execute("SELECT count(*) FROM message").fetchone()[0], 2)
        finally:
            check.close()


if __name__ == "__main__":
    unittest.main()
