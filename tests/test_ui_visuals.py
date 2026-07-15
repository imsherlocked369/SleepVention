import matplotlib.pyplot as plt

from src.ui_visuals import create_contribution_chart


def test_chart_supports_factors_and_empty_input():
    figure = create_contribution_chart(
        [{"display_name": "restlessness", "shap_value": -2.1}],
        [{"display_name": "REM sleep", "shap_value": 1.2}],
    )
    assert figure.axes[0].get_title() == "Main model contributions"
    assert figure.axes[0].get_xlabel() == "Contribution to predicted efficiency"
    plt.close(figure)
    empty = create_contribution_chart([], [])
    assert empty is not None
    plt.close(empty)
