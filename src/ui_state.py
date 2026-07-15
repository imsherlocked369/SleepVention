"""UI state helpers that can be tested without importing Streamlit."""

from pathlib import Path


def model_is_available(model_path: Path) -> bool:
    """Report model availability without loading or fabricating a model."""
    return Path(model_path).is_file()
