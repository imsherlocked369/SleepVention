"""Focused Matplotlib visuals for Streamlit result displays."""

from __future__ import annotations

import matplotlib.pyplot as plt


def create_contribution_chart(
    lowering: list[dict[str, object]],
    increasing: list[dict[str, object]],
    max_factors: int = 8,
):
    """Create a simple signed horizontal SHAP contribution chart."""
    factors = sorted(
        [*lowering, *increasing],
        key=lambda item: abs(float(item.get("shap_value", 0.0))),
        reverse=True,
    )[:max_factors]
    figure, axis = plt.subplots(figsize=(8, max(2.5, len(factors) * 0.55)))
    figure.patch.set_alpha(0)
    axis.set_facecolor("#23211e")
    if not factors:
        axis.text(0.5, 0.5, "No model contributions available", ha="center", va="center")
        axis.set_yticks([])
    else:
        factors.reverse()
        values = [float(item["shap_value"]) for item in factors]
        labels = [str(item.get("display_name", item.get("feature", "feature"))) for item in factors]
        colors = ["#86a873" if value >= 0 else "#d97757" for value in values]
        axis.barh(labels, values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Contribution to predicted efficiency")
    axis.set_title("Main model contributions")
    axis.xaxis.label.set_color("#b8b0a5")
    axis.title.set_color("#f3eee7")
    axis.tick_params(colors="#b8b0a5")
    for spine in axis.spines.values():
        spine.set_color("#3a3631")
    figure.tight_layout()
    return figure
