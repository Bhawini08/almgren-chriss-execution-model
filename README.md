# Almgren-Chriss Optimal Execution & Transaction Cost Model

A focused implementation of the Almgren-Chriss optimal liquidation framework for studying the trade-off between market impact and execution risk.

## Implements

- Inventory trajectory
- Trading trajectory
- Temporary market impact `eta`
- Permanent market impact `gamma`
- Volatility
- Risk aversion `lambda`
- Execution horizon
- Expected execution cost
- Execution-cost variance
- Efficient execution frontier
- Implementation shortfall
- TWAP benchmark
- Monte Carlo validation of analytical cost moments
- Parameter sensitivity analysis

## Validation snapshot

Using the default stylized parameter set:

- Expected execution cost: **$2.625m**
- Expected cost: **525 bps**
- Temporary-impact cost: **$2.500m**
- Permanent-impact cost: **$0.125m**
- Analytical cost stdev: **$11.98k**
- Simulated mean shortfall: **$2.625m**
- Simulated shortfall stdev: **$11.90k**
- Urgency parameter `kappa`: **0.0127**
- TWAP expected cost: **$2.625m**
- TWAP cost stdev: **$11.98k**

The Monte Carlo simulation closely matches the analytical mean and variance, which provides a direct model-consistency check.

The default risk aversion is intentionally low, so the optimized schedule remains close to TWAP. Increasing `lambda` shifts execution earlier, reducing risk at the cost of more temporary impact.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python scripts/run_analysis.py
streamlit run dashboard/app.py
```

The default analysis uses stylized parameters for validation. Intraday calibration is a natural extension but is not required to demonstrate the execution framework correctly.

## Scope discipline

This project intentionally does not claim low-latency execution, order-book reconstruction, HJB PDE control, or proprietary impact calibration. The objective is a transparent and testable implementation of the Almgren-Chriss framework.
