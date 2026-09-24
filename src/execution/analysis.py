from dataclasses import replace
import numpy as np
import pandas as pd
from .model import ACParams, AlmgrenChriss

def efficient_frontier(base: ACParams, lambdas=None):
    lambdas=np.logspace(-9,-2,40) if lambdas is None else np.asarray(lambdas,float)
    rows=[]
    for lam in lambdas:
        m=AlmgrenChriss(replace(base,risk_aversion=float(lam)))
        rows.append({"lambda":lam,"expected_cost":m.expected_cost(),"variance":m.cost_variance(),
                     "stdev_cost":np.sqrt(m.cost_variance()),"objective":m.objective()})
    return pd.DataFrame(rows)

def sensitivity(base: ACParams, field: str, values):
    rows=[]
    for v in values:
        p=replace(base,**{field:v}); m=AlmgrenChriss(p)
        first_half=m.trading_trajectory().iloc[:max(1,p.periods//2)].sum()/p.shares
        rows.append({field:v,"expected_cost":m.expected_cost(),"stdev_cost":np.sqrt(m.cost_variance()),
                     "first_half_fraction":first_half})
    return pd.DataFrame(rows)
