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

**Provenance & privacy:** the trajectory/patch/requirements files are verbatim
research-corpus artifacts from the DECAF evaluation on public SWE-bench Verified
tasks (agent runs produced for the DECAF paper's released artifact). They contain
run-structural identifiers (session UUIDs, sandbox paths, provider ids, model
reasoning text) that the adapters parse — sanitizing them would silently change
read-set/diagnosis behavior, so they are kept verbatim and **checked for
credential-shaped strings** by `tests/test_attribution.py::
test_fixture_contains_no_secrets`. Do not add fixture content from non-public
runs.

Fixture contents must stay byte-stable: the golden tests pin the expected
diagnosis as literals, and the identity check in `attribution.diagnose()`
compares displayed files byte-for-byte against `corpus/data/trajectory/...`.
