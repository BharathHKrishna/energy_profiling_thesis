"""Shared filename-safe slugging for stratum names — was independently reimplemented in
detection_map.py (x2), segmap_generator.py (x2), and audit_stream_output.py before
2026-08-11. All 5 copies were functionally identical (verified byte-for-byte before
consolidating), but audit_stream_output.py's whole job is to look up files the two map
generators wrote — a silent drift between copies would show up as false "missing file"
reports, not an error.
"""


def slug_name(name: str) -> str:
    """Turn a stratum name into a safe filename fragment, e.g.
    "Industrial + Water" -> "Industrial_plus_Water"."""
    return (name.replace(" ", "_").replace("/", "-")
                .replace("+", "plus").replace("(", "").replace(")", ""))
