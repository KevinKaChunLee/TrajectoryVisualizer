# Test fixtures

## `corpus/` + `decaf_cache/`

A minimal, hermetic corpus for the DECAF failure-attribution integration tests
(wired in `tests/conftest.py`): two SWE-bench Verified instances of the
`claude_code` agent run —

- `astropy__astropy-13033` — deductive-only diagnosis (primary
  `code_editing/incorrect_patch`)
- `django__django-11477` — arbiter-refuted case (`refuted_unattributed`), with
  its judge + arbiter verdicts vendored under `decaf_cache/` (the
  `z-ai__glm-5.2` cache namespace)

**Provenance & privacy:** the trajectory/patch/requirements files are
research-corpus artifacts from the DECAF evaluation on public SWE-bench Verified
tasks (agent runs produced for the DECAF paper's released artifact), **sanitized
before committing**: session/message UUIDs and request ids are replaced with
deterministic placeholders, signature/encrypted-content blobs are emptied, and
local absolute paths are rewritten to `/home/user`. The golden tests verify the
sanitized trajectories produce the exact same diagnosis (the stripped fields are
not consumed by the adapters). Fixtures are additionally **checked for
credential-shaped strings** by `tests/test_fixture_hygiene.py` (redacted
reporting). Do not add fixture content from non-public runs. For any new
fixture case, sanitize it manually with the same three transformations listed
above (deterministic placeholder UUIDs/request ids, emptied signature blobs,
paths rewritten to `/home/user`), then restamp provenance: recompute
`trajectory_sha256` over the sanitized canonical trajectory (and
`requirements_sha256` over its requirements file) and update those fields in
every vendored verdict record under `decaf_cache/` so the provenance check in
`attribution.py` still passes.

Fixture contents must stay byte-stable: the golden tests pin the expected
diagnosis as literals, and the identity check in `attribution.diagnose()`
compares displayed files byte-for-byte against `corpus/data/trajectory/...`.
