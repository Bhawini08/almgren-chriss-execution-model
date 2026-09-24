import json
from pathlib import Path
import numpy as np
from execution.model import ACParams, AlmgrenChriss
from execution.analysis import efficient_frontier, sensitivity, compare_to_twap, monte_carlo_validation

out=Path("results")
out.mkdir(exist_ok=True)

p=ACParams()
m=AlmgrenChriss(p)

m.schedule().to_csv(out/"optimal_schedule.csv",index=False)
efficient_frontier(p).to_csv(out/"efficient_frontier.csv",index=False)
sensitivity(p,"risk_aversion",np.logspace(-9,-3,25)).to_csv(out/"lambda_sensitivity.csv",index=False)
sensitivity(p,"sigma",np.linspace(0.005,0.06,20)).to_csv(out/"volatility_sensitivity.csv",index=False)
sensitivity(p,"eta",np.logspace(-7,-4,20)).to_csv(out/"impact_sensitivity.csv",index=False)
compare_to_twap(p).to_csv(out/"schedule_comparison.csv",index=False)
monte_carlo_validation(p).to_csv(out/"monte_carlo_validation.csv",index=False)

sf=m.simulate_shortfall(n_sims=20000)
metrics={
    "expected_cost":m.expected_cost(),
    "expected_cost_bps":m.expected_shortfall_bps(),
    "temporary_impact_cost":m.temporary_impact_cost(),
    "permanent_impact_cost":m.permanent_impact_cost(),
    "spread_cost":m.spread_cost(),
    "cost_stdev":float(np.sqrt(m.cost_variance())),
    "simulated_mean_shortfall":float(sf.mean()),
    "simulated_stdev_shortfall":float(sf.std(ddof=1)),
    "kappa":m.kappa,
    "twap_expected_cost":m.twap_expected_cost(),
    "twap_cost_stdev":float(np.sqrt(m.twap_cost_variance()))
}
(out/"metrics.json").write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
