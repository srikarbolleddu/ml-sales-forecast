"""Turn the training reports into the ml-demo-kit/v1 `/explore` payload: generic chart specs that
the shared ExplorePanel renders. Captions pull their numbers from the reports, so they can never
disagree with the charts."""

from __future__ import annotations

from typing import Any


def build_explore(explore: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["test"]
    best_naive = metrics["best_naive"]
    naive = metrics["comparison"][best_naive]
    leak = metrics["random_split"]
    leak_gain = 1 - leak["wape_random_split_model"] / leak["wape_time_split_model"]
    real_gain = metrics["improvement_vs_best_naive"]
    months = explore["wape_by_month"]
    peak = max(explore["monthly_units"], key=lambda m: m["value"])
    trough = min(explore["monthly_units"], key=lambda m: m["value"])

    return {
        "contract": "ml-demo-kit/v1",
        "headline": [
            {
                "label": "Forecast error (WAPE)",
                "value": test["wape"],
                "format": "percent",
                "hint": f"vs {naive['wape']:.1%} for “{best_naive.removeprefix('Naive: ')}”",
            },
            {
                "label": "Total forecast bias",
                "value": test["bias"],
                "format": "percent",
                "hint": f"naive: {naive['bias']:+.0%} (under-forecasts the Q4 ramp)",
            },
            {
                "label": "Typical miss",
                "value": test["mae_units"],
                "format": "number",
                "hint": "units per product per month (MAE)",
            },
            {
                "label": "Products",
                "value": metrics["dataset"]["products"],
                "format": "integer",
                "hint": "in the monthly panel",
            },
        ],
        "charts": [
            {
                "id": "split",
                "kind": "bar",
                "title": "Why the split has to follow time",
                "caption": (
                    f"The same model, trained two ways and scored on the same {leak['rows']:,} "
                    "held-out rows. Trained on a random split, it has seen months after the ones "
                    f"it forecasts, and looks {leak_gain:.0%} more accurate. That flattery is "
                    f"about {leak_gain / real_gain:.0f}x the model's real gain over the naive "
                    "forecast. Every other number here uses the honest, time-ordered split."
                    if real_gain > 0
                    else "The same model, trained on a random vs a time-ordered split."
                ),
                "format": "percent",
                "items": explore["split_comparison"],
            },
            {
                "id": "models",
                "kind": "table",
                "title": "Gradient boosting vs naive forecasts",
                "caption": (
                    "Error on the three held-out months (WAPE: total absolute error ÷ total units "
                    f"sold; lower is better). Beating “{best_naive.removeprefix('Naive: ')}” is "
                    f"hard: the model wins by {real_gain:.0%}. Its bigger advantage is bias: its "
                    f"total forecast is off by {test['bias']:+.0%}, the naive rule's by "
                    f"{naive['bias']:+.0%}."
                ),
                "columns": [
                    {"key": "model", "label": "Forecaster"},
                    {"key": "wape", "label": "WAPE", "format": "percent"},
                    {"key": "mae_units", "label": "MAE (units)", "format": "number"},
                    {"key": "bias", "label": "Bias", "format": "percent"},
                ],
                "rows": explore["model_comparison"],
                "highlight": {"key": "model", "value": "Gradient boosting"},
            },
            {
                "id": "by-month",
                "kind": "table",
                "title": "Forecast error by month",
                "caption": (
                    "The model beats the naive forecast in every held-out month."
                    if all(m["model_wape"] < m["naive_wape"] for m in months)
                    else "Model vs naive forecast error, month by month."
                ),
                "columns": [
                    {"key": "month", "label": "Month"},
                    {"key": "model_wape", "label": "Model WAPE", "format": "percent"},
                    {"key": "naive_wape", "label": "Naive WAPE", "format": "percent"},
                ],
                "rows": months,
            },
            {
                "id": "seasonality",
                "kind": "bar",
                "title": "Units sold per month (products in the panel)",
                "caption": (
                    f"Sales peak in {peak['label']} ({peak['value']:,} units) and bottom out in "
                    f"{trough['label']} ({trough['value']:,}). That is why the forecast month is "
                    "an input."
                ),
                "format": "integer",
                "items": explore["monthly_units"],
            },
            {
                "id": "importance",
                "kind": "bar",
                "title": "What the forecast relies on",
                "caption": (
                    "How much the typical miss grows (in units) when each input is shuffled "
                    "(permutation importance, held-out months)."
                ),
                "format": "number",
                "items": [
                    {"label": f["label"], "value": f["importance"]}
                    for f in explore["feature_importance"]
                    if f["importance"] > 0
                ],
            },
        ],
    }
