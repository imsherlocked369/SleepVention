"""Warm-dark terminal theme and safe reusable HTML for SleepVention."""

from __future__ import annotations

from html import escape

DESIGN_TOKENS = {
    "bg": "#181715",
    "bg_secondary": "#1e1c1a",
    "panel": "#23211e",
    "panel_raised": "#292622",
    "panel_soft": "#201e1b",
    "text": "#f3eee7",
    "text_secondary": "#b8b0a5",
    "text_muted": "#8f887f",
    "accent": "#d97757",
    "accent_hover": "#e28767",
    "success": "#86a873",
    "warning": "#d6a44f",
    "error": "#c96f6f",
    "border": "#3a3631",
    "border_soft": "#302c28",
}

SLEEPVENTION_CSS = r"""
<style>
:root {
    --sv-bg: #181715;
    --sv-bg-secondary: #1e1c1a;
    --sv-panel: #23211e;
    --sv-panel-raised: #292622;
    --sv-panel-soft: #201e1b;
    --sv-text: #f3eee7;
    --sv-text-secondary: #b8b0a5;
    --sv-text-muted: #8f887f;
    --sv-accent: #d97757;
    --sv-accent-hover: #e28767;
    --sv-accent-soft: rgba(217, 119, 87, 0.14);
    --sv-success: #86a873;
    --sv-warning: #d6a44f;
    --sv-error: #c96f6f;
    --sv-border: #3a3631;
    --sv-border-soft: #302c28;
    --sv-radius: 10px;
    --sv-font: "SFMono-Regular", "Cascadia Code", "Roboto Mono",
        "Liberation Mono", Menlo, Monaco, Consolas, monospace;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--sv-bg);
    color: var(--sv-text);
}
[data-testid="stHeader"] { background: rgba(24, 23, 21, 0.92); }
[data-testid="stMainBlockContainer"] { max-width: 1400px; padding-top: 1.8rem; }
[data-testid="stSidebar"] {
    background: var(--sv-bg-secondary);
    border-right: 1px solid var(--sv-border-soft);
}
[data-testid="stSidebar"] * { font-family: var(--sv-font); }

h1, h2, h3, label, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    font-family: var(--sv-font) !important;
}
p, li, .stMarkdown { color: var(--sv-text-secondary); }
code, pre { font-family: var(--sv-font) !important; }

.sv-header {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: 1.15rem 1.25rem; margin-bottom: 0.75rem;
    border: 1px solid var(--sv-border); border-radius: var(--sv-radius);
    background: var(--sv-panel);
}
.sv-title { font: 600 2.15rem/1.05 var(--sv-font); color: var(--sv-text); }
.sv-title span { color: var(--sv-accent); }
.sv-subtitle { margin-top: 0.42rem; color: var(--sv-text-muted); font: 0.86rem var(--sv-font); }
.sv-system-status { white-space: nowrap; color: var(--sv-text-secondary); font: 0.8rem var(--sv-font); }
.sv-status-dot { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; margin-right: 0.45rem; }
.sv-ready { background: var(--sv-success); }
.sv-limited { background: var(--sv-warning); }
.sv-unavailable { background: var(--sv-error); }

.sv-status-strip {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0;
    margin-bottom: 1.2rem; border: 1px solid var(--sv-border-soft);
    border-radius: var(--sv-radius); overflow: hidden; background: var(--sv-panel-soft);
}
.sv-status-item { min-width: 0; padding: 0.78rem 0.9rem; border-right: 1px solid var(--sv-border-soft); }
.sv-status-item:last-child { border-right: 0; }
.sv-label { color: var(--sv-text-muted); font: 0.7rem var(--sv-font); letter-spacing: 0.11em; text-transform: uppercase; }
.sv-value { margin-top: 0.32rem; overflow-wrap: anywhere; color: var(--sv-text); font: 0.84rem var(--sv-font); }

.sv-section {
    margin: 1.5rem 0 0.8rem; padding: 0.68rem 0.85rem;
    border: 1px solid var(--sv-border-soft); border-left: 3px solid var(--sv-accent);
    border-radius: var(--sv-radius); background: var(--sv-panel-soft);
    color: var(--sv-text); font: 600 0.78rem var(--sv-font);
    letter-spacing: 0.1em; text-transform: uppercase;
}
.sv-console {
    padding: 0.9rem 1rem; border: 1px solid var(--sv-border);
    border-radius: var(--sv-radius); background: var(--sv-panel);
    font: 0.82rem/1.7 var(--sv-font); color: var(--sv-text-secondary);
}
.sv-log-ok { color: var(--sv-success); }
.sv-log-warn { color: var(--sv-warning); }
.sv-log-error { color: var(--sv-error); }
.sv-log-info { color: var(--sv-text-secondary); }
.sv-empty {
    padding: 1rem; border: 1px dashed var(--sv-border); border-radius: var(--sv-radius);
    background: var(--sv-panel-soft); color: var(--sv-text-muted); font: 0.84rem var(--sv-font);
}

[data-testid="stFileUploader"] {
    padding: 0.8rem; border: 1px dashed var(--sv-border);
    border-radius: var(--sv-radius); background: var(--sv-panel);
}
[data-testid="stDataFrame"], [data-testid="stMetric"] {
    border: 1px solid var(--sv-border-soft); border-radius: var(--sv-radius);
    background: var(--sv-panel);
}
[data-testid="stMetric"] { padding: 0.85rem 1rem; }
[data-testid="stMetricValue"] { color: var(--sv-text) !important; font-size: 1.45rem !important; }

.stButton > button[kind="primary"] {
    border: 1px solid var(--sv-accent); background: var(--sv-accent);
    color: #181715; font: 600 0.88rem var(--sv-font); box-shadow: none;
}
.stButton > button[kind="primary"]:hover {
    border-color: var(--sv-accent-hover); background: var(--sv-accent-hover); color: #181715;
}
.stButton > button, .stDownloadButton > button {
    border-color: var(--sv-border); background: var(--sv-panel-raised);
    color: var(--sv-text); font-family: var(--sv-font); box-shadow: none;
}
.stDownloadButton > button:hover, .stButton > button:not([kind="primary"]):hover {
    border-color: var(--sv-accent); color: var(--sv-accent-hover);
}
button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {
    outline: 2px solid var(--sv-accent-hover) !important; outline-offset: 2px;
}

[data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea {
    background: var(--sv-panel) !important; border-color: var(--sv-border) !important;
    color: var(--sv-text) !important; font-family: var(--sv-font) !important;
}
[data-testid="stExpander"] { border-color: var(--sv-border-soft); background: var(--sv-panel-soft); }
[data-baseweb="tab-list"] { gap: 1rem; border-bottom: 1px solid var(--sv-border-soft); }
[data-baseweb="tab"] { color: var(--sv-text-muted); font-family: var(--sv-font); }
[aria-selected="true"][data-baseweb="tab"] { color: var(--sv-text); border-bottom-color: var(--sv-accent); }
[data-testid="stProgress"] > div > div { background: var(--sv-accent); }
hr { border-color: var(--sv-border-soft); }

@media (max-width: 800px) {
    .sv-header { align-items: flex-start; flex-direction: column; }
    .sv-status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .sv-status-item:nth-child(2) { border-right: 0; }
    .sv-status-item:nth-child(-n+2) { border-bottom: 1px solid var(--sv-border-soft); }
    .sv-title { font-size: 1.8rem; }
}
@media (max-width: 480px) {
    .sv-status-strip { grid-template-columns: 1fr; }
    .sv-status-item { border-right: 0; border-bottom: 1px solid var(--sv-border-soft); }
}
</style>
"""


def apply_sleepvention_theme() -> None:
    """Inject the complete local CSS theme as one organized block."""
    import streamlit as st

    st.markdown(SLEEPVENTION_CSS, unsafe_allow_html=True)


def terminal_header_html(
    title: str, subtitle: str, status: str, status_kind: str = "ready"
) -> str:
    """Build an escaped terminal header."""
    safe_kind = status_kind if status_kind in {"ready", "limited", "unavailable"} else "limited"
    return (
        '<div class="sv-header"><div>'
        f'<div class="sv-title">{escape(title)}<span>_</span></div>'
        f'<div class="sv-subtitle">{escape(subtitle)}</div></div>'
        '<div class="sv-system-status">'
        f'<span class="sv-status-dot sv-{safe_kind}"></span>{escape(status)}</div></div>'
    )


def status_strip_html(items: list[tuple[str, str]]) -> str:
    """Build escaped terminal metadata cards."""
    blocks = "".join(
        '<div class="sv-status-item">'
        f'<div class="sv-label">{escape(label)}</div>'
        f'<div class="sv-value">{escape(value)}</div></div>'
        for label, value in items
    )
    return f'<div class="sv-status-strip">{blocks}</div>'


def section_heading_html(number: str, label: str) -> str:
    return f'<div class="sv-section">{escape(number)} / {escape(label)}</div>'


def console_log_html(entries: list[tuple[str, str]]) -> str:
    """Build an escaped console log from ok/warn/error/info entries."""
    allowed = {"ok", "warn", "error", "info"}
    lines = []
    for level, message in entries:
        safe_level = level if level in allowed else "info"
        lines.append(
            f'<div class="sv-log-{safe_level}">[{safe_level}] {escape(message)}</div>'
        )
    return '<div class="sv-console">' + "".join(lines) + "</div>"


def empty_state_html(message: str) -> str:
    return f'<div class="sv-empty">&gt; {escape(message)}</div>'
