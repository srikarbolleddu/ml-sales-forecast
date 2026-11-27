from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.features import FEATURE_ORDER, TARGET, build_panel, clean_transactions, records_to_frame


def _line(invoice="500000", code="85123A", qty=10, price=2.5, date="2011-01-15", customer=1.0):
    return {
        "Invoice": invoice,
        "StockCode": code,
        "Quantity": qty,
        "Price": price,
        "InvoiceDate": date,
        "Customer ID": customer,
    }


def test_clean_transactions_drops_cancellations_returns_and_non_products():
    raw = pd.DataFrame(
        [
            _line(),
            _line(invoice="C500001"),  # cancellation
            _line(qty=-3),  # return
            _line(price=0),  # free item
            _line(code="POST"),  # postage, not a product
        ]
    )

    sales = clean_transactions(raw)

    assert len(sales) == 1
    assert str(sales.iloc[0]["month"]) == "2011-01"


def _sales(units_by_month: dict[str, int], code="10001", price=2.0):
    return clean_transactions(
        pd.DataFrame(
            [
                _line(code=code, qty=u, price=price, date=f"{m}-10")
                for m, u in units_by_month.items()
            ]
        )
    )


def test_panel_rows_use_only_the_past_and_target_the_next_month():
    sales = _sales({"2011-01": 5, "2011-02": 7, "2011-03": 9, "2011-04": 11})

    panel = build_panel(sales, "2011-04")
    row = panel[panel["month"].astype(str) == "2011-03"].iloc[0]

    assert row["units_last_month"] == 9
    assert row["units_2_months_ago"] == 7
    assert row["units_3_months_ago"] == 5
    assert row[TARGET] == 11
    assert row["forecast_month"] == "Apr"
    assert row["months_on_sale"] == 3


def test_same_month_last_year_is_missing_without_a_year_of_history():
    sales = _sales({"2010-01": 4, "2010-12": 6, "2011-01": 8})

    panel = build_panel(sales, "2011-01")
    dec_row = panel[panel["month"].astype(str) == "2010-12"].iloc[0]
    first_row = panel.iloc[0]

    assert dec_row["units_same_month_last_year"] == 4  # Jan 2011 vs Jan 2010
    assert np.isnan(first_row["units_same_month_last_year"])


def test_dormant_products_are_not_forecast():
    sales = _sales({"2011-01": 5, "2011-06": 5})

    months = build_panel(sales, "2011-06")["month"].astype(str).tolist()

    assert "2011-04" not in months  # no sales in the three months up to April
    assert "2011-03" in months


def test_price_change_is_relative_to_three_months_earlier():
    sales = pd.concat(
        [
            _sales({"2011-01": 5}, price=2.0),
            _sales({"2011-02": 5, "2011-03": 5}, price=2.0),
            _sales({"2011-04": 5, "2011-05": 5}, price=3.0),
        ]
    )

    panel = build_panel(sales, "2011-05")
    row = panel[panel["month"].astype(str) == "2011-04"].iloc[0]  # Apr price vs Jan price

    assert row["price_change_pct"] == pytest.approx(50.0)


def test_records_to_frame_keeps_missing_optional_values():
    record: dict[str, Any] = dict.fromkeys(FEATURE_ORDER, 1.0) | {"forecast_month": "Nov"}
    record["units_same_month_last_year"] = None

    frame = records_to_frame([record])

    assert list(frame.columns) == FEATURE_ORDER
    assert np.isnan(frame.loc[0, "units_same_month_last_year"])
