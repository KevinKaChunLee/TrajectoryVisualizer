"""Bridge module: orchestrates Converge's comparison pipeline from Insight's UI.

Delegates to converge.alignment.build_comparison_report_from_steps — the same
core the file-path entry point (build_comparison_report) uses — so the two
pipelines cannot drift (R21). This module only adapts already-loaded trajectory
dicts, parses the optional anchor patch via the shared helper, and renders the
HTML report.
"""

import traceback

from trajviz.insight.parser import parse_steps

from trajviz.converge.alignment import (
    DEFAULT_TOKEN_RATE,
    _parse_anchor_files,
    build_comparison_report_from_steps,
)
from trajviz.converge.rendering import build_comparison_report_html


def run_comparison(
    ref_raw: dict,
    cmp_raw: dict,
    anchor_path: str | None = None,
    token_rate: float = DEFAULT_TOKEN_RATE,
    fuzzy: bool = False,
    dark: bool = False,
) -> dict:
    """Run Converge's full comparison pipeline.

    Parameters
    ----------
    ref_raw : dict
        Raw trajectory dict for the **reference / baseline** trajectory —
        the one the user uploaded on the Comparison tab (left of the anchor
        patch slot). Already loaded via ``load_trajectory``.
    cmp_raw : dict
        Raw trajectory dict for the **compared** trajectory — the one the
        user loaded on the Overview tab. Already loaded via ``load_trajectory``.
    anchor_path : str or None
        Optional path to a .patch/.diff file for anchor-grounded comparison.
    token_rate : float
        Token rate for cost computation (default 50.0).
    fuzzy : bool
        Enable fuzzy command matching.
    dark : bool
        Unused (the HTML report is theme-agnostic); retained for API
        stability with the Insight caller.

    Returns
    -------
    dict with keys:
        report_html : str
            The rendered comparison report (or an error banner on failure).
        ok : bool
            True when the comparison ran to completion; False when a
            trajectory failed to load or the pipeline raised (B15's
            producer side — the caller branches its status line on this).
    """
    empty = {"report_html": "", "ok": False}

    try:
        if "_error" in ref_raw:
            empty["report_html"] = (
                f"<div style='color:var(--ov-bad);padding:1em;'>"
                f"Error loading reference trajectory: {ref_raw['_error']}</div>"
            )
            return empty
        if "_error" in cmp_raw:
            empty["report_html"] = (
                f"<div style='color:var(--ov-bad);padding:1em;'>"
                f"Error loading compared trajectory: {cmp_raw['_error']}</div>"
            )
            return empty

        ref_steps = parse_steps(ref_raw)
        cmp_steps = parse_steps(cmp_raw)

        report = build_comparison_report_from_steps(
            ref_raw, cmp_raw, ref_steps, cmp_steps,
            token_rate=token_rate,
            fuzzy_commands=fuzzy,
            anchor_files=_parse_anchor_files(anchor_path),
            ref_path=ref_raw.get("_source_path", ""),
            cmp_path=cmp_raw.get("_source_path", ""),
        )

        # Render HTML report. The Insight UI's Comparison tab suppresses the
        # divergence-patterns section (heading, glossary, table) — the patterns
        # are still computed for downstream consumers but are hidden from the
        # rendered report. The standalone Converge app (converge/app.py) goes
        # through build_comparison_report directly and is unaffected.
        report_html = build_comparison_report_html({**report, "patterns": []})

        return {"report_html": report_html, "ok": True}

    except Exception as e:
        empty["report_html"] = (
            f"<div style='color:var(--ov-bad);padding:1em;'>"
            f"<strong>Comparison failed:</strong> {type(e).__name__}: {e}"
            f"<pre style='font-size:11px;margin-top:8px;color:var(--ov-muted);'>"
            f"{traceback.format_exc()}</pre></div>"
        )
        return empty
