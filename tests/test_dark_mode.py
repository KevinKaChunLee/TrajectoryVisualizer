"""Tests for dark mode CSS variable completeness and rendering.py inline colors."""

import re

import pytest


class TestDarkModeCSSVariables:
    """Verify that key :root CSS variables have dark mode overrides."""

    def _get_root_vars(self):
        """Extract variable names from the :root block."""
        from trajectory_visualizer.insight.styles import APP_CSS

        root_match = re.search(r':root\s*\{([^}]+)\}', APP_CSS)
        assert root_match, ":root block not found in APP_CSS"
        root_block = root_match.group(1)
        return set(re.findall(r'(--[\w-]+):', root_block))

    def _get_dark_vars(self):
        """Extract variable names from the dark media query :root block."""
        from trajectory_visualizer.insight.styles import APP_CSS

        dark_match = re.search(
            r'@media\s*\(prefers-color-scheme:\s*dark\)\s*\{[^}]*:root\s*\{([^}]+)\}',
            APP_CSS, re.DOTALL
        )
        assert dark_match, "Dark mode :root block not found in APP_CSS"
        dark_block = dark_match.group(1)
        return set(re.findall(r'(--[\w-]+):', dark_block))

    def test_core_ov_variables_have_dark_overrides(self):
        """All --ov-* variables defined in :root must have dark overrides."""
        root_vars = self._get_root_vars()
        dark_vars = self._get_dark_vars()

        ov_vars = {v for v in root_vars if v.startswith("--ov-")}
        missing = ov_vars - dark_vars
        assert not missing, f"Missing dark overrides for: {sorted(missing)}"

    def test_core_wf_variables_have_dark_overrides(self):
        """All --wf-* variables defined in :root must have dark overrides."""
        root_vars = self._get_root_vars()
        dark_vars = self._get_dark_vars()

        wf_vars = {v for v in root_vars if v.startswith("--wf-")}
        missing = wf_vars - dark_vars
        assert not missing, f"Missing dark overrides for: {sorted(missing)}"


class TestRenderingNoHardcodedColors:
    """Verify rendering.py doesn't use hardcoded light-only colors in inline styles."""

    def test_no_hardcoded_color_in_inline_styles(self):
        """Grep for hardcoded hex colors in rendering.py inline style attributes."""
        import inspect
        from trajectory_visualizer.insight import rendering

        source = inspect.getsource(rendering)

        # Match inline style color/background with hardcoded hex
        pattern = r"(?:color|background|background-color)\s*:\s*#[0-9a-fA-F]{3,8}"
        matches = re.findall(pattern, source)

        # Filter out colors that are OK in dark mode (used in non-inline contexts)
        # Allow colors in CSS class definitions (not inline styles)
        # We specifically check f-string inline styles
        inline_pattern = r"""style\s*=\s*['"][^'"]*(?:color|background|background-color)\s*:\s*#[0-9a-fA-F]{3,8}"""
        inline_matches = re.findall(inline_pattern, source)

        assert not inline_matches, (
            f"Found {len(inline_matches)} hardcoded color(s) in inline styles:\n"
            + "\n".join(inline_matches[:5])
        )
