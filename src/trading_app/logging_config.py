from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure app logging once for CLI, Streamlit, and tests."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
