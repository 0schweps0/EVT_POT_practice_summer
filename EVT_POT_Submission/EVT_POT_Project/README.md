# EVT/POT Tail-Risk Internship Project

## Research question

The project applies the Peaks-over-Threshold (POT) branch of Extreme Value Theory to daily financial losses. It estimates the Generalized Pareto Distribution (GPD), computes one-day EVT VaR and Expected Shortfall at 99% and 99.5%, and validates rolling forecasts with the Kupiec and Christoffersen tests.

## Assets and sample

A preliminary screening of ten liquid US stocks selected **NFLX, META and INTC** as the three heaviest-tailed series. The market benchmark is the **S&P 500**. The common estimation sample contains 1,153 daily loss observations from 29 July 2021 to 3 March 2026. Loss is defined as the negative logarithmic return.

## Data sources

- Candidate US stocks: the public `misterdonn/finance-datasets` repository on Hugging Face, containing Yahoo Finance / yfinance-style daily OHLCV files.
- S&P 500: the public `nadtoka/predictive-stock-dataset` repository on Hugging Face.

The raw CSV files used in the report are preserved in `data/raw/` for reproducibility.

## Project structure

```text
EVT_POT_Project/
├── data/raw/                 # raw public market data
├── notebooks/
│   └── EVT_POT_Analysis.ipynb
├── src/
│   └── evt_pot_analysis.py   # complete reusable analysis module
├── results/
│   ├── figures/              # publication-ready charts
│   └── tables/               # all numerical outputs as CSV
├── requirements.txt
└── README.md
```

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/evt_pot_analysis.py
```

The report calculations use 500 parametric-bootstrap replications for each Cramer-von Mises test and 1,000 bootstrap replications for confidence intervals. A quick functional run can be made with:

```bash
python src/evt_pot_analysis.py --cvm-reps 50 --risk-reps 100 --skip-backtest
```

## Main model choices

| Asset | POT threshold | Exceedances | Shape xi | Scale beta | CvM bootstrap p-value |
|---|---:|---:|---:|---:|---:|
| NFLX | 92nd percentile | 93 | 0.411 | 0.0128 | 0.766 |
| META | 95th percentile | 58 | 0.723 | 0.0086 | 0.679 |
| INTC | 92nd percentile | 93 | 0.250 | 0.0179 | 0.493 |
| S&P 500 | 92nd percentile | 93 | 0.124 | 0.0068 | 0.768 |

All point estimates of xi are positive, implying heavy right tails in the loss distribution. META has the heaviest fitted tail and the greatest ES uncertainty.

## Main empirical conclusions

- EVT estimates show that VaR alone does not capture tail severity: META has a particularly large and uncertain Expected Shortfall because its fitted shape parameter is high.
- Threshold sensitivity is limited for NFLX, INTC and the S&P 500. META is the exception: 90% and 92% thresholds fail the GPD goodness-of-fit test, so the 95% threshold is required.
- Rolling backtests use a 600-day window and 501 forecasts from 4 March 2024 to 3 March 2026.
- Kupiec unconditional coverage is not rejected for any asset/level.
- Conditional coverage is not rejected in seven of eight cases. The exception is S&P 500 at 99.5%, where violations are clustered and the Christoffersen conditional-coverage p-value is 0.040.

