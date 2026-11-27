"""Feature definitions and data preparation, shared by training (`training/train.py`) and serving.

The SuperKart notebook's goal (forecast product sales from product and store attributes with a
gradient-boosting regressor), rebuilt on real, openly licensed retail transactions. Raw invoice
lines become a monthly panel: one row per product per month, describing what was known at the end
of that month, with next month's units as the target.

Dataset: Online Retail II (Chen, 2019), UCI ML Repository, CC BY 4.0. A UK online gift retailer's
transactions, December 2009 - December 2011.
https://archive.ics.uci.edu/dataset/502/online+retail+ii
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

TARGET = "units_next_month"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class Field:
    """One model input. Drives training columns, request validation and the demo form."""

    name: str
    label: str
    kind: Literal["number", "category"]
    group: Literal["history", "product", "timing"]
    help: str
    source_column: str
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str | None = None
    optional: bool = False
    option_labels: dict[str, str] = field(default_factory=dict)

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "group": self.group,
            "help": self.help,
            "optional": self.optional,
        }
        if self.kind == "category":
            schema["options"] = [
                {"value": value, "label": self.option_labels.get(value, _humanize(value))}
                for value in self.options
            ]
        else:
            schema.update(
                {"min": self.minimum, "max": self.maximum, "step": self.step, "unit": self.unit}
            )
        return schema


def _humanize(value: str) -> str:
    return value.replace(".", " ").replace("-", " ").replace("_", " ").strip().capitalize()


FIELDS: tuple[Field, ...] = (
    Field(
        "units_last_month",
        "Units sold last month",
        "number",
        "history",
        "Units of this product sold in the most recent full month.",
        "units_0",
        minimum=0,
        maximum=20_000,
        step=1,
        unit="units",
    ),
    Field(
        "units_2_months_ago",
        "Units sold 2 months ago",
        "number",
        "history",
        "Units sold the month before last.",
        "units_1",
        minimum=0,
        maximum=20_000,
        step=1,
        unit="units",
    ),
    Field(
        "units_3_months_ago",
        "Units sold 3 months ago",
        "number",
        "history",
        "Units sold three months ago.",
        "units_2",
        minimum=0,
        maximum=20_000,
        step=1,
        unit="units",
    ),
    Field(
        "units_same_month_last_year",
        "Units sold in that month last year",
        "number",
        "history",
        "Units sold in the forecast month one year earlier. Leave empty if the "
        "product wasn't on sale then.",
        "units_lag_12",
        minimum=0,
        maximum=20_000,
        step=1,
        unit="units",
        optional=True,
    ),
    Field(
        "customers_last_month",
        "Customers last month",
        "number",
        "history",
        "Distinct customers who bought the product last month.",
        "customers_0",
        minimum=0,
        maximum=1_000,
        step=1,
        unit="customers",
    ),
    Field(
        "unit_price",
        "Unit price",
        "number",
        "product",
        "Typical selling price per unit (last known).",
        "price",
        minimum=0.01,
        maximum=100,
        step=0.01,
        unit="GBP",
    ),
    Field(
        "price_change_pct",
        "Price change over 3 months",
        "number",
        "product",
        "Change in typical price compared with three months earlier.",
        "price_change_pct",
        minimum=-90,
        maximum=300,
        step=0.1,
        unit="%",
    ),
    Field(
        "months_on_sale",
        "Months on sale",
        "number",
        "product",
        "Months since the product's first sale (in this dataset).",
        "age",
        minimum=1,
        maximum=24,
        step=1,
        unit="months",
    ),
    Field(
        "forecast_month",
        "Forecast month",
        "category",
        "timing",
        "The month being forecast. Gift sales peak before Christmas.",
        "target_month",
        options=MONTHS,
        option_labels={m: m for m in MONTHS},
    ),
)

FIELDS_BY_NAME = {f.name: f for f in FIELDS}
NUMERIC_FEATURES = [f.name for f in FIELDS if f.kind == "number"]
CATEGORICAL_FEATURES = [f.name for f in FIELDS if f.kind == "category"]
FEATURE_ORDER = [f.name for f in FIELDS]

# Stock codes for postage, fees and manual adjustments start with a letter; products with digits.
PRODUCT_CODE = r"^\d{5}"


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Invoice lines → sales lines: drop cancellations, returns, zero prices and non-products."""
    df = raw.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    keep = (
        ~df["Invoice"].str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & df["StockCode"].str.match(PRODUCT_CODE)
    )
    df = df[keep]
    df["month"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("M")
    return df[["StockCode", "month", "Quantity", "Price", "Customer ID"]]


def monthly_sales(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.groupby(["StockCode", "month"])
        .agg(
            units=("Quantity", "sum"),
            customers=("Customer ID", "nunique"),
            price=("Price", "median"),
        )
        .reset_index()
    )


def build_panel(sales: pd.DataFrame, last_month: str) -> pd.DataFrame:
    """One row per product and month t (≤ last_month - 1): features known at the end of t, and
    the target = units sold in t + 1. Only products sold in at least one of the last 3 months are
    forecast (dormant products are a different problem)."""
    monthly = monthly_sales(sales)
    end = pd.Period(last_month, "M")
    months = pd.period_range(monthly["month"].min(), end, freq="M")
    rows = []
    for code, group in monthly.groupby("StockCode"):
        series = group.set_index("month").reindex(months)
        units = series["units"].fillna(0).to_numpy()
        customers = series["customers"].fillna(0).to_numpy()
        price = series["price"].ffill().to_numpy()
        first = int(np.argmax(units > 0))
        for i in range(first, len(months) - 1):
            recent = units[max(i - 2, 0) : i + 1]
            if recent.sum() == 0:
                continue
            target_month = months[i + 1]
            lag12 = i + 1 - 12
            price_then = price[i - 3] if i >= 3 and not np.isnan(price[i - 3]) else np.nan
            rows.append(
                {
                    "StockCode": code,
                    "month": months[i],
                    "target_month_period": target_month,
                    "units_last_month": units[i],
                    "units_2_months_ago": units[i - 1] if i >= 1 else 0.0,
                    "units_3_months_ago": units[i - 2] if i >= 2 else 0.0,
                    # Missing if the product wasn't on sale yet a year earlier (or the data
                    # doesn't reach back that far), as opposed to 0 = on sale but none sold.
                    "units_same_month_last_year": units[lag12] if lag12 >= first else np.nan,
                    "customers_last_month": customers[i],
                    "unit_price": price[i],
                    "price_change_pct": (price[i] / price_then - 1) * 100
                    if price_then and not np.isnan(price_then)
                    else 0.0,
                    "months_on_sale": i - first + 1,
                    "forecast_month": MONTHS[target_month.month - 1],
                    TARGET: units[i + 1],
                }
            )
    panel = pd.DataFrame(rows)
    for name in NUMERIC_FEATURES:
        f = FIELDS_BY_NAME[name]
        panel[name] = panel[name].clip(lower=f.minimum, upper=f.maximum)
    return panel.dropna(subset=["unit_price"]).reset_index(drop=True)


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """API inputs → the feature columns the model was trained on."""
    frame = pd.DataFrame(
        [{name: record.get(name) for name in FEATURE_ORDER} for record in records],
        columns=pd.Index(FEATURE_ORDER),
    )
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    return frame
