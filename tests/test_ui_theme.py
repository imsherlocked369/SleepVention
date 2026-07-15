import sys
from types import SimpleNamespace

from src.ui_theme import (
    DESIGN_TOKENS,
    SLEEPVENTION_CSS,
    apply_sleepvention_theme,
    console_log_html,
    empty_state_html,
    section_heading_html,
    status_strip_html,
    terminal_header_html,
)


def test_theme_contains_expected_local_design_tokens():
    assert DESIGN_TOKENS["bg"] == "#181715"
    assert DESIGN_TOKENS["accent"] == "#d97757"
    assert "--sv-panel: #23211e" in SLEEPVENTION_CSS
    assert "--sv-success: #86a873" in SLEEPVENTION_CSS
    assert "http://" not in SLEEPVENTION_CSS
    assert "https://" not in SLEEPVENTION_CSS
    assert "anthropic" not in SLEEPVENTION_CSS.lower()
    assert "claude" not in SLEEPVENTION_CSS.lower()


def test_dynamic_content_is_html_escaped():
    dangerous = '<img src=x onerror="alert(1)">'
    rendered = "".join(
        [
            terminal_header_html(dangerous, dangerous, dangerous),
            status_strip_html([(dangerous, dangerous)]),
            section_heading_html(dangerous, dangerous),
            console_log_html([("error", dangerous)]),
            empty_state_html(dangerous),
        ]
    )
    assert dangerous not in rendered
    assert "&lt;img" in rendered


def test_apply_theme_renders_single_css_block(monkeypatch):
    calls = []
    fake_streamlit = SimpleNamespace(
        markdown=lambda value, unsafe_allow_html=False: calls.append(
            (value, unsafe_allow_html)
        )
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    apply_sleepvention_theme()
    assert calls == [(SLEEPVENTION_CSS, True)]
