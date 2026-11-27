"""Load the trained forecaster and explain each forecast with what-if contributions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio
from sklearn.pipeline import Pipeline

from app.features import FEATURE_ORDER, FIELDS_BY_NAME, NUMERIC_FEATURES, records_to_frame

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.skops"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
SAMPLES_PATH = ARTIFACT_DIR / "samples.json"

# skops refuses to load any type not explicitly trusted. These are exactly the non-default types
# this Pipeline(ColumnTransformer + OrdinalEncoder + HistGradientBoostingRegressor) serializes. The
# numeric "passthrough" is stored as functools.partial(check_array, ...); skops checks the wrapped
# function separately, so trusting `partial` doesn't trust arbitrary code. Anything else means the
# file is not the artifact we trained, so loading fails closed.
TRUSTED_TYPES = frozenset(
    {
        "functools.partial",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        "sklearn.utils.validation.check_array",
    }
)

# How many inputs to show in an explanation (the largest effects either way).
TOP_CONTRIBUTIONS = 6


def save(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    sio.dump(pipeline, path)


def load(path: Path = MODEL_PATH) -> Pipeline:
    untrusted = set(sio.get_untrusted_types(file=path))
    unexpected = untrusted - TRUSTED_TYPES
    if unexpected:
        raise RuntimeError(f"refusing to load model with unexpected types: {sorted(unexpected)}")
    return sio.load(path, trusted=sorted(untrusted))


def _value_text(name: str, value: Any) -> str:
    field = FIELDS_BY_NAME[name]
    if field.kind == "category":
        option = next(o for o in field.to_schema()["options"] if o["value"] == value)
        return f"{field.label.rstrip('?')}: {option['label']}"
    if value is None:
        return f"{field.label}: not available"
    number = float(value)
    if field.unit == "GBP":
        return f"{field.label}: £{number:,.2f}"
    digits = 0 if float(field.step or 1).is_integer() else 1  # match the form's precision
    text = f"{number:,.{digits}f}" + (f" {field.unit}" if field.unit else "")
    return f"{field.label}: {text}"


class SalesModel:
    def __init__(self, pipeline: Pipeline, manifest: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.version: str = manifest["model_version"]
        self.baseline: dict[str, Any] = manifest["baseline"]
        self.threshold = None  # a forecast has no decision cutoff

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        """Next-month units plus what-if contributions.

        For each input, re-forecast with just that input set to a typical value (training median /
        most common month). The difference is that input's contribution in units: positive = this
        value raises the forecast. One batched call.
        """
        rows = [record]
        for name in FEATURE_ORDER:
            rows.append({**record, name: self.baseline[name]})
        forecasts = np.clip(self.pipeline.predict(records_to_frame(rows)), 0, None)
        forecast = float(forecasts[0])

        contributions: list[dict[str, Any]] = []
        for name, without in zip(FEATURE_ORDER, forecasts[1:], strict=True):
            effect = forecast - float(without)
            if self._is_baseline(name, record[name]) or abs(effect) < 0.5:
                continue
            contributions.append(
                {"label": _value_text(name, record[name]), "field": name, "value": round(effect, 1)}
            )
        contributions.sort(key=lambda c: -abs(c["value"]))

        units = round(forecast)
        return {
            "task": "regression",
            "score": units,
            "score_label": "Forecast for next month",
            "unit": "units",
            "threshold": None,
            "positive": None,
            "outcome": f"About {units:,} units",
            "explanation": {
                "kind": "contributions",
                "title": "What moved the forecast",
                "value_label": (
                    "Units each input adds to (+) or removes from (−) the forecast, compared with "
                    "a typical product"
                ),
                "steps": contributions[:TOP_CONTRIBUTIONS],
            },
            "model_version": self.version,
        }

    def _is_baseline(self, name: str, value: Any) -> bool:
        base = self.baseline[name]
        if name in NUMERIC_FEATURES:
            if value is None or base is None:
                return value is base
            return float(value) == float(base)
        return value == base


@lru_cache(maxsize=1)
def get_model() -> SalesModel:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return SalesModel(load(MODEL_PATH), manifest)


@lru_cache(maxsize=1)
def get_samples() -> list[dict[str, Any]]:
    return json.loads(SAMPLES_PATH.read_text())
