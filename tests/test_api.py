from app.core.config import settings


def test_health(client):
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_schema_lists_every_field_with_groups(client):
    body = client.get("/api/v1/schema").json()

    assert body["contract"] == "ml-demo-kit/v1"
    assert {g["id"] for g in body["groups"]} == {"history", "product", "timing"}
    assert len(body["fields"]) == 9
    assert body["task"] == "regression"


def test_samples_cover_both_outcomes(client):
    samples = client.get("/api/v1/samples").json()

    assert all(s["actual"].endswith(" sold") for s in samples)


def test_explore_has_chart_data(client):
    body = client.get("/api/v1/explore").json()

    kinds = {chart["kind"] for chart in body["charts"]}
    assert kinds == {"bar", "table"}
    assert all(chart["title"] and chart["caption"] for chart in body["charts"])
    assert len(body["headline"]) == 4


def test_predict_returns_score_and_contributions(client, sample_inputs):
    body = client.post("/api/v1/predict", json=sample_inputs).json()

    assert body["score"] >= 0 and body["unit"] == "units"
    assert body["threshold"] is None and body["positive"] is None
    assert body["explanation"]["kind"] == "contributions"
    steps = body["explanation"]["steps"]
    assert all(step["label"] and step["value"] != 0 for step in steps)
    assert [abs(s["value"]) for s in steps] == sorted(
        (abs(s["value"]) for s in steps), reverse=True
    )


def test_typical_product_has_no_contributions(client):
    from app.model import get_model

    body = client.post("/api/v1/predict", json=get_model().baseline).json()

    assert body["explanation"]["steps"] == []


def test_predict_rejects_unknown_category(client, sample_inputs):
    sample_inputs["forecast_month"] = "Smarch"

    assert client.post("/api/v1/predict", json=sample_inputs).status_code == 422


def test_predict_rejects_out_of_range_number(client, sample_inputs):
    sample_inputs["unit_price"] = 5000

    assert client.post("/api/v1/predict", json=sample_inputs).status_code == 422


def test_predict_rejects_extra_fields(client, sample_inputs):
    sample_inputs["units_next_month"] = 10  # the target itself must never be an input

    assert client.post("/api/v1/predict", json=sample_inputs).status_code == 422


def test_oversized_body_is_rejected_before_parsing(client):
    response = client.post(
        "/api/v1/predict",
        content=b"x" * (settings.MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


def test_rate_limit_returns_429(client, sample_inputs, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)

    codes = [client.post("/api/v1/predict", json=sample_inputs).status_code for _ in range(3)]

    assert codes == [200, 200, 429]


def test_cors_allows_portfolio_origin_only(client):
    allowed = client.get("/api/v1/health", headers={"Origin": "https://srikarbolleddu.com"})
    other = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "https://srikarbolleddu.com"
    assert "access-control-allow-origin" not in other.headers
