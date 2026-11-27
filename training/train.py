"""Reproducible training: download the pinned dataset → monthly product panel → compare forecasters
on a time-ordered split → write artifacts.

    uv run python training/train.py

Writes:
  app/model_artifacts/model.skops, manifest.json, samples.json, metrics.json, explore.json
  reports/metrics.json      held-out metrics -- the only source for numbers quoted anywhere
  reports/explore.json      precomputed data for the demo's interactive "Explore" charts
  reports/golden.json       fixed inputs + expected outputs, checked by tests/test_golden.py
  reports/figures/*.png     charts for the README / case study / blog

Method follows the UT Austin PGP-AIML SuperKart project (gradient-boosting regression of product
sales), rebuilt on the openly licensed Online Retail II transactions.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import model as model_io  # noqa: E402
from app.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    FIELDS,
    MONTHS,
    NUMERIC_FEATURES,
    TARGET,
    build_panel,
    clean_transactions,
    records_to_frame,
)

DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
CACHE = ROOT / ".data" / "online_retail_ii.zip"
# Parsing the 45 MB workbook takes a couple of minutes; cache the cleaned sales lines as CSV.
SALES_CACHE = ROOT / ".data" / "sales_lines.csv"
LAST_FULL_MONTH = "2011-11"  # the data ends on 2011-12-09, so December is partial
TRAIN_END, VALIDATION_END = "2011-06", "2011-08"  # by forecast month; test = Sep-Nov 2011
SEED = 42
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def fetch_sales() -> pd.DataFrame:
    if SALES_CACHE.exists():
        sales = pd.read_csv(SALES_CACHE, dtype={"StockCode": str})
        sales["month"] = pd.PeriodIndex(sales["month"], freq="M")
        return sales
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        # DATA_URL is a constant https URL and the payload is checksum-verified below.
        with urllib.request.urlopen(DATA_URL, timeout=300) as response:  # nosec B310
            CACHE.write_bytes(response.read())
    payload = CACHE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise SystemExit(f"dataset checksum mismatch: {digest} != {DATA_SHA256}")
    print("reading workbook (a couple of minutes)")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        workbook = archive.read("online_retail_II.xlsx")
    sheets = pd.read_excel(io.BytesIO(workbook), sheet_name=None, engine="openpyxl")
    sales = clean_transactions(pd.concat(sheets.values(), ignore_index=True))
    sales.to_csv(SALES_CACHE, index=False)
    return sales


def build_pipeline(loss: str, learning_rate: float, max_leaf_nodes: int) -> Pipeline:
    encoder = OrdinalEncoder(categories=[list(MONTHS)])
    prep = ColumnTransformer(
        [("cat", encoder, CATEGORICAL_FEATURES), ("num", "passthrough", NUMERIC_FEATURES)]
    )
    model = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=400,
        categorical_features=[True] + [False] * len(NUMERIC_FEATURES),
        random_state=SEED,
    )
    return Pipeline([("prep", prep), ("model", model)])


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error / total actual units."""
    return float(np.abs(actual - forecast).sum() / actual.sum())


def evaluate(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    return {
        "wape": round(wape(actual, forecast), 4),
        "mae_units": round(float(np.abs(actual - forecast).mean()), 2),
        "bias": round(float(forecast.sum() / actual.sum() - 1), 4),
        "rows": len(actual),
    }


def baselines(x: pd.DataFrame) -> dict[str, np.ndarray]:
    recent = x[["units_last_month", "units_2_months_ago", "units_3_months_ago"]]
    seasonal = x["units_same_month_last_year"].fillna(x["units_last_month"])
    return {
        "Naive: same as last month": x["units_last_month"].to_numpy(),
        "Naive: 3-month average": recent.mean(axis=1).to_numpy(),
        "Naive: same month last year": seasonal.to_numpy(),
    }


def main() -> None:
    panel = build_panel(fetch_sales(), LAST_FULL_MONTH)
    month = panel["target_month_period"].astype(str)
    train = panel[month <= TRAIN_END]
    val = panel[(month > TRAIN_END) & (month <= VALIDATION_END)]
    test = panel[month > VALIDATION_END]
    x_train, y_train = train[FEATURE_ORDER], train[TARGET].to_numpy()
    x_val, y_val = val[FEATURE_ORDER], val[TARGET].to_numpy()
    x_test, y_test = test[FEATURE_ORDER], test[TARGET].to_numpy()
    print(f"rows: train={len(train)} val={len(val)} test={len(test)} (split by forecast month)")

    # Model choice on validation WAPE. Poisson loss suits counts; squared error is the default.
    best, best_wape = None, float("inf")
    best_params: dict[str, Any] = {}
    for loss in ("poisson", "squared_error"):
        for learning_rate, leaves in ((0.05, 31), (0.1, 31), (0.05, 63)):
            candidate = build_pipeline(loss, learning_rate, leaves).fit(x_train, y_train)
            score = wape(y_val, np.clip(candidate.predict(x_val), 0, None))
            print(f"  {loss} lr={learning_rate} leaves={leaves}: val WAPE {score:.4f}")
            if score < best_wape:
                best, best_wape = candidate, score
                best_params = {
                    "loss": loss,
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": leaves,
                }
    if best is None:
        raise RuntimeError("no model trained")

    # Refit the chosen settings on train + validation, then touch the test months once.
    dev = pd.concat([train, val])
    final = build_pipeline(**best_params).fit(dev[FEATURE_ORDER], dev[TARGET].to_numpy())
    test_forecast = np.clip(final.predict(x_test), 0, None)

    comparison = {"Gradient boosting": {**evaluate(y_test, test_forecast), **best_params}}
    for name, forecast in baselines(x_test).items():
        comparison[name] = evaluate(y_test, forecast)
    best_naive = min(
        (n for n in comparison if n.startswith("Naive")), key=lambda n: comparison[n]["wape"]
    )
    improvement = 1 - comparison["Gradient boosting"]["wape"] / comparison[best_naive]["wape"]

    # Does a random split flatter the model? Train the same settings on a random 80% of ALL rows
    # (so it sees months after the test rows), then score both models on the same rows: test-month
    # rows that neither model trained on. Same rows, so the two errors are directly comparable.
    x_all, y_all = panel[FEATURE_ORDER], panel[TARGET].to_numpy()
    random_train_idx, random_test_idx = train_test_split(
        panel.index, test_size=0.2, random_state=SEED
    )
    leaky = build_pipeline(**best_params).fit(x_all.loc[random_train_idx], y_all[random_train_idx])
    shared = test.index.intersection(random_test_idx)
    shared_actual = panel.loc[shared, TARGET].to_numpy()
    random_split = {
        "rows": len(shared),
        "wape_random_split_model": round(
            wape(shared_actual, np.clip(leaky.predict(panel.loc[shared, FEATURE_ORDER]), 0, None)),
            4,
        ),
        "wape_time_split_model": round(
            wape(shared_actual, np.clip(final.predict(panel.loc[shared, FEATURE_ORDER]), 0, None)),
            4,
        ),
    }

    test_months = month.loc[test.index].to_numpy()
    by_month = []
    for m in sorted(set(test_months)):
        mask = test_months == m
        label = f"{MONTHS[int(m[-2:]) - 1]} {m[:4]}"
        naive_forecast = x_test["units_last_month"].to_numpy()[mask]
        by_month.append(
            {
                "month": label,
                "model_wape": round(wape(y_test[mask], test_forecast[mask]), 4),
                "naive_wape": round(wape(y_test[mask], naive_forecast), 4),
            }
        )

    perm: Any = permutation_importance(
        final, x_test, y_test, scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED
    )
    labels = {f.name: f.label for f in FIELDS}
    importance_rows = sorted(
        (
            {"field": n, "label": labels[n], "importance": round(float(v), 3)}
            for n, v in zip(FEATURE_ORDER, perm.importances_mean, strict=True)
        ),
        key=lambda r: -float(r["importance"]),
    )

    baseline: dict[str, Any] = {n: float(dev[n].median()) for n in NUMERIC_FEATURES}
    baseline.update({n: str(dev[n].mode().iloc[0]) for n in CATEGORICAL_FEATURES})

    params_hash = hashlib.sha256(str(final.get_params()).encode()).hexdigest()[:8]
    model_version = f"gbm-{best_params['loss'].replace('_', '-')}-{params_hash}"
    artifacts = model_io.ARTIFACT_DIR
    artifacts.mkdir(parents=True, exist_ok=True)
    model_io.save(final, model_io.MODEL_PATH)
    manifest = {
        "model_version": model_version,
        "model": "Gradient boosting regressor",
        "params": best_params,
        "baseline": baseline,
        "dataset": {"url": DATA_URL, "sha256": DATA_SHA256, "license": "CC BY 4.0"},
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Real held-out product-months, chosen deterministically to show different situations.
    t = test.assign(forecast=test_forecast, error=np.abs(test_forecast - y_test))
    t = t.sort_values(["units_last_month", "StockCode"])
    nov = t[t.forecast_month == "Nov"]
    lags = t[["units_last_month", "units_2_months_ago", "units_3_months_ago"]]
    stable_price = t.price_change_pct.abs() < 5
    wanted = [
        ("Consistent best seller in November", nov[lags.loc[nov.index].min(axis=1) > 400], 0.5),
        (
            "Christmas item ramping up",
            nov[nov.units_same_month_last_year > 3 * nov.units_last_month.clip(lower=1)],
            0.5,
        ),
        (
            "Steady small seller",
            t[lags.min(axis=1).ge(10) & lags.max(axis=1).le(40) & stable_price],
            0.5,
        ),
        ("New product (not on sale last year)", t[t.units_same_month_last_year.isna()], 0.5),
        ("After a price cut", t[t.price_change_pct < -20], 0.5),
        ("A forecast that missed badly", t.sort_values("error"), 0.99),
    ]
    used: set[Any] = set()
    samples: list[dict[str, Any]] = []
    for title, group, position in wanted:
        free = group.drop(index=[i for i in group.index if i in used])
        if free.empty:
            continue
        row = free.iloc[min(int(len(free) * position), len(free) - 1)]
        used.add(row.name)
        inputs: dict[str, Any] = {}
        for name in FEATURE_ORDER:
            value = row[name]
            if name in NUMERIC_FEATURES:
                if pd.isna(value):
                    value = None
                else:
                    step = (
                        0.01 if name == "unit_price" else 0.1 if name == "price_change_pct" else 1
                    )
                    value = round(round(float(value) / step) * step, 2)
                    value = int(value) if float(value).is_integer() else value
            inputs[name] = value
        samples.append(
            {
                "title": title,
                "inputs": inputs,
                "actual": f"{int(row[TARGET]):,} unit{'' if int(row[TARGET]) == 1 else 's'} sold",
            }
        )
    (artifacts / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    REPORTS.mkdir(exist_ok=True)
    metrics = {
        "dataset": {
            "name": "UCI Online Retail II (Dec 2009 - Nov 2011, full months)",
            "product_months": len(panel),
            "products": int(panel["StockCode"].nunique()),
            "split": {
                "train": f"forecast months to {TRAIN_END} ({len(train)} rows)",
                "validation": f"to {VALIDATION_END} ({len(val)} rows)",
                "test": f"Sep-Nov 2011 ({len(test)} rows)",
            },
        },
        "final_model": "Gradient boosting",
        "params": best_params,
        "test": comparison["Gradient boosting"],
        "best_naive": best_naive,
        "improvement_vs_best_naive": round(improvement, 4),
        "comparison": comparison,
        "random_split": random_split,
        "model_version": model_version,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    explore = {
        "model_comparison": [{"model": n, **c} for n, c in comparison.items()],
        "split_comparison": [
            {
                "label": "Trained on a random split (saw later months)",
                "value": random_split["wape_random_split_model"],
            },
            {
                "label": "Trained only on earlier months",
                "value": random_split["wape_time_split_model"],
            },
        ],
        "wape_by_month": by_month,
        "feature_importance": importance_rows,
        "monthly_units": [
            {"label": f"{MONTHS[m.month - 1]} {m.year}", "value": int(v)}
            for m, v in panel.groupby("target_month_period")[TARGET].sum().items()
        ],
    }
    (REPORTS / "explore.json").write_text(json.dumps(explore, indent=2) + "\n")

    golden = [
        {
            "inputs": s["inputs"],
            "forecast": round(float(final.predict(records_to_frame([s["inputs"]]))[0]), 4),
        }
        for s in samples
    ]
    (REPORTS / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    for name in ("metrics.json", "explore.json"):
        (artifacts / name).write_text((REPORTS / name).read_text())

    write_figures(comparison, importance_rows)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ("test", "best_naive", "improvement_vs_best_naive", "random_split")
            },
            indent=2,
        )
    )


def write_figures(comparison: dict[str, Any], importance: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(comparison)
    values = [c["wape"] * 100 for c in comparison.values()]
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("WAPE on Sep-Nov 2011 (%) — lower is better")
    ax.set_title("Forecast error: models vs naive baselines")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = importance[::-1]
    ax.barh([r["label"] for r in top], [r["importance"] for r in top])
    ax.set_xlabel("increase in mean absolute error when shuffled (units)")
    ax.set_title("What the forecast relies on")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
