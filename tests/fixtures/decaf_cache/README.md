# Vendored DECAF verdict caches

Two real LLM verdicts — one judge, one arbiter — for
`claude_code / django__django-11477`, vendored so the attribution integration
tests run hermetically: no developer-local caches, no API key, no network, and
no dependence on DECAF's own (gitignored) cache directories.

`conftest.py` points DECAF's judge and arbiter cache directories here, forces the
judge-model namespace to `z-ai/glm-5.2` (which is the namespace these files live
under), and points `config.PROJECT_DATA` at this directory so DECAF's repo-local
inputs (`pertest/`, `overrides/`) resolve as absent instead of reaching outside
the fixture tree.

## Why these files carry `"evidence_schema": 20`

`attribution._verdict_verifies` trusts a cached verdict only when its full prompt
provenance still matches the current inputs: `trajectory_sha256`,
`requirements_sha256`, `prompt_version` and `evidence_schema`. A verdict whose
provenance cannot be verified is ignored and its layer is disabled — that refusal
is the security property, so the stamps in these files must never be edited just
to make a test pass.

These two were produced under `evidence_schema: 11`. DECAF later moved
`SCHEMA_VERSION` to 20, which disabled both layers. The stamps were re-derived to
20 only after proving the model's input did not change — i.e. that these verdicts
are still verdicts *on exactly the same prompt*:

* `judge.build_messages` was rendered for this case under both the schema-11 and
  the schema-20 code against this fixture tree, and the two system+user prompts
  are **byte-identical** (user prompt sha256 `12a54db57230…`, 4089 chars, and the
  evidence dict has the same keys under both).
* `arbiter.build_messages` was rendered the same way for the recorded claim
  (`code_editing` / `relevant_change_omitted`) and is likewise **byte-identical**
  (4454 chars).
* `prompt_version` did not move for either layer (judge `6511d1a70c39`, arbiter
  `f7f11661e443`), so only the schema counter changed.

To re-verify, render `build_messages` for this case under the two schema versions
and diff. If a future schema bump *does* change the rendered prompt, re-stamping
is no longer sound: regenerate the verdicts, or let the layers be refused and
adjust the tests to expect refusal.
