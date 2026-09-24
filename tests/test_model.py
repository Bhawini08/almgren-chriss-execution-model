from dataclasses import replace
import numpy as np
from execution.model import ACParams, AlmgrenChriss
from execution.analysis import efficient_frontier

def test_schedule_liquidates_inventory():
    m=AlmgrenChriss(ACParams())
    x=m.inventory_trajectory(); n=m.trading_trajectory()
    assert np.isclose(x.iloc[0],m.p.shares)
    assert abs(x.iloc[-1])<1e-6
    assert np.isclose(n.sum(),m.p.shares)
    assert (n>=-1e-8).all()

def test_more_risk_aversion_frontloads_execution():
    p=ACParams()
    lo=AlmgrenChriss(replace(p,risk_aversion=1e-9)).trading_trajectory()
    hi=AlmgrenChriss(replace(p,risk_aversion=1e-2)).trading_trajectory()
    assert hi.iloc[:5].sum() > lo.iloc[:5].sum()

def test_frontier_has_finite_cost_and_risk():
    f=efficient_frontier(ACParams(),[1e-9,1e-6,1e-3])
    assert np.isfinite(f[["expected_cost","variance"]]).all().all()
    assert (f.expected_cost>0).all() and (f.variance>=0).all()
