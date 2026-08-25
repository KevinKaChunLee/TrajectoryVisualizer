"""Object-payload format sniffing shared by parse and detect."""

_OBJECT_FORMATS = frozenset({"ccsession", "codearts", "opencode"})
_EVENT_FORMATS = frozenset({"codex", "pi", "dsh"})
_FORMAT_STAMPS = {
    "ccsession": "_cc_format",
    "codearts": "_codearts_format",
    "codex": "_codex_format",
    "pi": "_pi_format",
    "dsh": "_dsh_format",
}

def _detect_object_format(raw: dict) -> str:
    """Detect format of a parsed JSON object (raw export or already-converted)."""
    # Post-conversion markers: converters build/stamp a dict so a second pass
    # (UI gate, attribution, run-group) still reports the originating product.
    if raw.get("_cc_format") is True:
        return "ccsession"
    if raw.get("_codex_format") is True:
        return "codex"
    if raw.get("_pi_format") is True:
        return "pi"
    if raw.get("_dsh_format") is True:
        return "dsh"
    if raw.get("format") == "ccsession-trajectory":
        return "ccsession"
    # CodeArts exports use an OpenCode-compatible ``info + messages``
    # envelope.  Check their explicit export marker before the generic
    # OpenCode shape so the UI does not mislabel the originating product.
    export_metadata = raw.get("export_metadata")
    if raw.get("_codearts_format") is True or (
        isinstance(export_metadata, dict)
        and export_metadata.get("schema_version") == 2
        and export_metadata.get("source_format") == "codearts_opencode_sqlite"
        and isinstance(raw.get("info"), dict)
        and isinstance(raw.get("messages"), list)
    ):
        return "codearts"
    # CodeArts legacy-JSON exports (codearts_consolidator.py emits
    # source_format "codearts_legacy_json") lack the OpenCode "info" dict but
    # are still CodeArts files — detect them so the UI's format-mismatch gate
    # and labeling work instead of falling through to "unknown".
    if (
        isinstance(export_metadata, dict)
        and export_metadata.get("schema_version") == 2
        and export_metadata.get("source_format") == "codearts_legacy_json"
        and isinstance(raw.get("messages"), list)
    ):
        return "codearts"
    if isinstance(raw.get("info"), dict) and isinstance(raw.get("messages"), list):
        return "opencode"
    return "unknown"
