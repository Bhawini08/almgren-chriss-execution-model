import json
from pathlib import Path
import numpy as np
from execution.model import ACParams, AlmgrenChriss
from execution.analysis import efficient_frontier, sensitivity
out=Path("results"); out.mkdir(exist_ok=True)
p=ACParams(); m=AlmgrenChriss(p)
m.schedule().to_csv(out/"optimal_schedule.csv",index=False)
efficient_frontier(p).to_csv(out/"efficient_frontier.csv",index=False)
sensitivity(p,"risk_aversion",np.logspace(-9,-3,25)).to_csv(out/"lambda_sensitivity.csv",index=False)
sensitivity(p,"sigma",np.linspace(0.005,0.06,20)).to_csv(out/"volatility_sensitivity.csv",index=False)
sensitivity(p,"eta",np.logspace(-7,-4,20)).to_csv(out/"impact_sensitivity.csv",index=False)
sf=m.simulate_shortfall()
metrics={"expected_cost":m.expected_cost(),"cost_stdev":float(np.sqrt(m.cost_variance())),
"simulated_mean_shortfall":float(sf.mean()),"simulated_stdev_shortfall":float(sf.std(ddof=1)),"kappa":m.kappa}
(out/"metrics.json").write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
