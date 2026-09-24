from dataclasses import replace
import numpy as np
from execution.model import ACParams, AlmgrenChriss
from execution.analysis import efficient_frontier, compare_to_twap, monte_carlo_validation

def test_schedule_liquidates_inventory():
    m=AlmgrenChriss(ACParams())
    x=m.inventory_trajectory(); n=m.trading_trajectory()
    assert np.isclose(x.iloc[0],m.p.shares)
    assert abs(x.iloc[-1])<1e-6
    assert np.isclose(n.sum(),m.p.shares)
    assert (n>=-1e-8).all()

def test_zero_risk_aversion_is_twap():
    p=ACParams(risk_aversion=0)
    m=AlmgrenChriss(p)
    n=m.trading_trajectory().to_numpy()
    assert np.allclose(n,np.repeat(p.shares/p.periods,p.periods),rtol=1e-8,atol=1e-6)

def test_more_risk_aversion_frontloads_execution():
    p=ACParams()
    lo=AlmgrenChriss(replace(p,risk_aversion=1e-9)).trading_trajectory()
    hi=AlmgrenChriss(replace(p,risk_aversion=1e-2)).trading_trajectory()
    assert hi.iloc[:5].sum() > lo.iloc[:5].sum()

def test_frontier_has_finite_cost_and_risk():
    f=efficient_frontier(ACParams(),[1e-9,1e-6,1e-3])
    assert np.isfinite(f[["expected_cost","variance","expected_cost_bps"]]).all().all()
    assert (f.expected_cost>0).all() and (f.variance>=0).all()

def test_ac_objective_no_worse_than_twap_for_positive_lambda():
    p=ACParams(risk_aversion=1e-3)
    x=compare_to_twap(p).set_index("schedule")
    assert x.loc["Almgren-Chriss","objective"] <= x.loc["TWAP","objective"] + 1e-6

def test_monte_carlo_matches_analytical_moments():
    p=ACParams()
    x=monte_carlo_validation(p,sims=[20000])
    row=x.iloc[0]
    assert abs(row["simulated_mean"]-row["analytical_mean"]) < 4*row["analytical_stdev"]/np.sqrt(20000)
    assert abs(row["simulated_stdev"]-row["analytical_stdev"])/row["analytical_stdev"] < 0.05
