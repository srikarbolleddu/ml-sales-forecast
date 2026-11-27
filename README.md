# Product Sales Forecast

How many units will this product sell next month? A **gradient-boosting forecaster** trained on
two years of real transactions from a UK online gift retailer, served as a **live demo API** with an
interactive UI. Change a product's recent sales, price or the month being forecast, and see the
forecast plus the inputs that moved it.

- **Live demo:** [srikarbolleddu.github.io/ml-sales-forecast](https://srikarbolleddu.github.io/ml-sales-forecast/)
  (standalone, from `web/`), also embedded in the
  [case study on srikarbolleddu.com](https://srikarbolleddu.com/projects/sales-forecast-prediction-system)
- **Stack:** scikit-learn (HistGradientBoostingRegressor, Poisson loss) · FastAPI · AWS Lambda
  (container) · API Gateway · React (Vite)

> Project from the UT Austin Post-Graduate Program in AI & Machine Learning (SuperKart), rebuilt from
> scratch on an openly licensed dataset. The course's notebooks and data are not used or included.

## Results (held-out months: Sep–Nov 2011, 9,315 product-months)

| Forecaster | WAPE | MAE (units) | Total bias |
|---|---|---|---|
| **Gradient boosting (Poisson loss)** | **52.4%** | **106.5** | **−4%** |
| Naive: same as last month | 54.0% | 109.8 | −15% |
| Naive: 3-month average | 54.3% | 110.3 | −26% |
| Naive: same month last year | 73.6% | 149.6 | +19% |

WAPE is total absolute error ÷ total units sold. All numbers come from
[`reports/metrics.json`](reports/metrics.json), which `training/train.py` writes.

**The honest headline:** the model beats "same as last month" by only **3%** on WAPE. It wins in
every held-out month, but monthly demand for individual products here is noisy (bulk wholesale
orders come and go), and last month's sales are a strong baseline. The bigger difference is
**bias**: the model's total forecast is 4% low over the three months, while the naive rule is 15%
low going into the Christmas ramp. For stock planning, that's the number that matters.

### Why the split follows time

Rows are split by the month being forecast: training on forecast months up to June 2011,
validation on July and August, and testing once on September to November. To measure what a random
split would have hidden, the same model was also trained on a random 80% of all rows, which includes
months *after* the ones it's tested on. Both models were then scored on the same 1,891 held-out
rows:

| Trained on | WAPE on the same rows |
|---|---|
| A random split (saw later months) | 48.0% |
| Earlier months only (honest) | 50.9% |

The random split makes the model look **6% better**, about twice its real gain over the naive
forecast. A random split would have turned a modest model into an impressive-looking one.

## Data preparation

`app/features.py` turns about 1M raw invoice lines into a **monthly panel**: one row per product per
month, holding only what was known at the end of that month, with next month's units as the target.

- Cancellations (`C…` invoices), returns, zero prices and non-product codes (postage, fees) are
  removed.
- Inputs: units sold in each of the last three months, units sold in the same month last year
  (*missing* if the product wasn't on sale yet), customers last month, typical price and its change
  over three months, months on sale, and the month being forecast.
- Only products sold in the last three months are forecast. Dormant products are a different
  problem.

## Using the API

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/schema` | Input fields with types, ranges, units and help text (drives the demo form) |
| `GET /api/v1/samples` | Six real held-out product-months, including a bad miss |
| `GET /api/v1/explore` | Chart specs: random vs time split, model vs naive, error by month, seasonality, importance |
| `POST /api/v1/predict` | Next-month forecast (units) and what-if contributions in units |
| `GET /api/v1/docs` | Interactive OpenAPI docs |

## Run locally

```bash
uv sync
uv run python training/train.py         # downloads the pinned dataset (sha256-checked) and retrains
uv run fastapi dev app/main.py          # API on http://localhost:8000/api/v1/docs
```

Demo page: `cd web && pnpm install && VITE_DEMO_API_URL=http://localhost:8000/api/v1 pnpm dev`
(without the variable it uses recorded responses in `web/src/fixture.json`; refresh them with
`uv run python scripts/export_web_fixture.py`, which a test enforces).

## Security and abuse limits (public demo)

- No secrets, and nothing visitors send is stored.
- Strict input validation (ranges, known categories, no extra fields) and a 4 KB body cap.
- Throttling: an API Gateway stage limit across all callers, plus a per-client rate limit in the app.
- Deploys use GitHub OIDC (no stored AWS keys). The deploy role trusts only this repo's
  approval-gated `production` environment.

## Data and license

- **Dataset:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
  (Chen, 2019), UCI Machine Learning Repository, CC BY 4.0. Downloaded at training time and
  checksum-pinned, not redistributed.
- **Code:** see [LICENSE](LICENSE).
