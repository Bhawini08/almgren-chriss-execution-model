from dataclasses import replace
import numpy as np
import pandas as pd
from .model import ACParams, AlmgrenChriss

def efficient_frontier(base: ACParams, lambdas=None):
    lambdas=np.logspace(-9,-2,40) if lambdas is None else np.asarray(lambdas,float)
    rows=[]
    for lam in lambdas:
        m=AlmgrenChriss(replace(base,risk_aversion=float(lam)))
        rows.append({
            "lambda":lam,
            "expected_cost":m.expected_cost(),
            "expected_cost_bps":m.expected_shortfall_bps(),
            "variance":m.cost_variance(),
            "stdev_cost":np.sqrt(m.cost_variance()),
            "objective":m.objective(),
            "first_half_fraction":m.trading_trajectory().iloc[:max(1,m.p.periods//2)].sum()/m.p.shares
        })
    return pd.DataFrame(rows)

def sensitivity(base: ACParams, field: str, values):
    rows=[]
    for v in values:
        p=replace(base,**{field:v}); m=AlmgrenChriss(p)
        first_half=m.trading_trajectory().iloc[:max(1,p.periods//2)].sum()/p.shares
        rows.append({
            field:v,
            "expected_cost":m.expected_cost(),
            "expected_cost_bps":m.expected_shortfall_bps(),
            "stdev_cost":np.sqrt(m.cost_variance()),
            "first_half_fraction":first_half
        })
    return pd.DataFrame(rows)

def compare_to_twap(base: ACParams):
    m=AlmgrenChriss(base)
    return pd.DataFrame([
        {
            "schedule":"Almgren-Chriss",
            "expected_cost":m.expected_cost(),
            "expected_cost_bps":m.expected_shortfall_bps(),
            "cost_stdev":np.sqrt(m.cost_variance()),
            "objective":m.objective(),
        },
        {
            "schedule":"TWAP",
            "expected_cost":m.twap_expected_cost(),
            "expected_cost_bps":m.twap_expected_cost()/(base.shares*base.price)*1e4,
            "cost_stdev":np.sqrt(m.twap_cost_variance()),
            "objective":m.twap_expected_cost()+base.risk_aversion*m.twap_cost_variance(),
        }
    ])

def monte_carlo_validation(base: ACParams, sims=(1000,5000,20000), seed=42):
    m=AlmgrenChriss(base); rows=[]
    analytical_mean=m.expected_cost()
    analytical_sd=np.sqrt(m.cost_variance())
    for n in sims:
        sf=m.simulate_shortfall(n_sims=int(n),seed=seed)
        rows.append({
            "simulations":int(n),
            "simulated_mean":float(sf.mean()),
            "analytical_mean":analytical_mean,
            "mean_error":float(sf.mean()-analytical_mean),
            "simulated_stdev":float(sf.std(ddof=1)),
            "analytical_stdev":analytical_sd,
            "stdev_error":float(sf.std(ddof=1)-analytical_sd),
        })
    return pd.DataFrame(rows)
