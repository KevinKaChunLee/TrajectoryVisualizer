# Training Label Reference

This document defines the canonical semantics for training-data label v2.

The design goal is to preserve the existing behavior labels (`phase` and
`action`) while adding a small, high-signal evaluation layer for assistant
turn quality, training value, and deterministic filtering decisions.

## Core Principles

- `phase` and `action` remain behavior labels only.
- `quality` evaluates whether a turn is locally trainable and clean.
- `value` evaluates whether a turn is worth retaining in a compressed,
  high-teaching-value SFT trajectory.
- `decision` is derived by deterministic rules, not directly authored by the
  labeling model.
- Value is **marginal teaching contribution**, not local answer quality.

## Labeling Unit

One assistant turn / assistant step is labeled at a time.

User turns and tool turns are context for judgment, not primary labeled rows in
the v2 output file.

## Quality

### verdict

Allowed values:

- `good`
- `usable`
- `flawed`
- `reject`

Interpretation:

- `good`: high-quality positive SFT candidate
- `usable`: acceptable, but not especially strong
- `flawed`: has notable issues but may still carry signal
- `reject`: should not be kept as positive SFT

### defect_flags

Allowed values:

- `incorrect`
- `unsupported_claim`
- `instruction_violation`
- `incomplete`
- `oververbose_noise`
- `unsafe_or_sensitive`
- `format_broken`
- `context_misread`

### confidence

Allowed values:

- `high`
- `medium`
- `low`

## Value

### tier

Allowed values:

- `high`
- `medium`
- `low`
- `none`

Interpretation:

- `high`: should usually survive trajectory compression
- `medium`: useful enough to retain in many cases
- `low`: weak training utility
- `none`: not worth keeping as positive SFT signal

### tags

Allowed values:

- `new_evidence_introduced`
- `strategy_pivot`
- `successful_recovery`
- `high_skill_operation`
- `verification_anchor`
- `reasoning_pattern`
- `tool_use_pattern`
- `negative_example`

### confidence

Allowed values:

- `high`
- `medium`
- `low`

## Why Value Is Not Local Quality

A turn can be locally reasonable but still be low value if:

- it introduces no new information
- it repeats a failed strategy without refinement
- a later successful turn does not depend on it
- it is administrative or tool-echo noise

A turn can be high value because it:

- introduced a decisive new fact
- pivoted the strategy correctly
- recovered from failure using prior evidence
- executed a relatively rare, high-skill operation
- anchored the final verification of success

## Decision Policy Inputs

`decision.label` is derived from:

- `quality.verdict`
- `quality.defect_flags`
- `quality.confidence`
- `value.tier`
- `value.tags`
- `value.confidence`

## Decision Labels

Allowed values:

- `keep`
- `drop`
- `review`

### Intended policy shape

- Keep when quality is positive and value is meaningful.
- Drop when quality is rejecting or value is clearly absent.
- Review when quality and value conflict, or confidence is low.

## Notes

- This file defines semantics and allowed values, not implementation details.
- Deterministic rule IDs live in the implementation layer, not here.
- Loss-mask policy is out of scope for v2 and deferred to v3.
