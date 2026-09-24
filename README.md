# Almgren-Chriss Optimal Execution & Transaction Cost Model

A focused implementation of the Almgren-Chriss optimal liquidation framework for studying the trade-off between market impact and execution risk.

## Implements

- Inventory trajectory
- Trading trajectory
- Temporary impact `eta`
- Permanent impact `gamma`
- Volatility
- Risk aversion `lambda`
- Execution horizon
- Expected execution cost
- Execution-cost variance
- Efficient execution frontier
- Implementation shortfall
- Parameter sensitivity analysis

The model intentionally stays within the scope of optimal execution rather than claiming order-book reconstruction, low-latency execution, or HJB-based control.

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

The default analysis uses stylized parameters for model validation. Intraday calibration can be added during the live-data pass.
