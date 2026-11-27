"""The shipped model must reproduce the outputs recorded at training time (reports/golden.json)."""

import json
from pathlib import Path

import pytest

from app.model import get_model

GOLDEN = json.loads((Path(__file__).resolve().parents[1] / "reports" / "golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: str(c["forecast"]))
def test_golden_prediction(case):
    # The API reports whole units; golden.json keeps the raw forecast.
    assert get_model().predict(case["inputs"])["score"] == round(max(case["forecast"], 0))


def test_model_refuses_unexpected_types(tmp_path, monkeypatch):
    import skops.io as sio

    from app import model as model_io

    path = tmp_path / "evil.skops"
    sio.dump({"fn": print}, path)
    monkeypatch.setattr(model_io, "TRUSTED_TYPES", frozenset())

    with pytest.raises(RuntimeError, match="unexpected types"):
        model_io.load(path)
