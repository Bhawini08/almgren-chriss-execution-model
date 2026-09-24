from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ACParams:
    shares: float = 1_000_000
    price: float = 50.0
    horizon: float = 1.0
    periods: int = 20
    sigma: float = 0.02
    eta: float = 2.5e-6
    gamma: float = 2.5e-7
    risk_aversion: float = 1e-6
    half_spread: float = 0.0

class AlmgrenChriss:
    def __init__(self, params: ACParams):
        if params.periods < 2 or params.horizon <= 0 or params.eta <= 0 or params.sigma < 0:
            raise ValueError("Invalid Almgren-Chriss parameters")
        if params.shares <= 0 or params.price <= 0 or params.gamma < 0 or params.risk_aversion < 0:
            raise ValueError("shares/price must be positive; impact/risk parameters non-negative")
        self.p=params
        self.tau=params.horizon/params.periods

    @property
    def eta_tilde(self) -> float:
        return float(self.p.eta - 0.5*self.p.gamma*self.tau)

    @property
    def kappa(self) -> float:
        if self.p.risk_aversion <= 0 or self.p.sigma == 0:
            return 0.0
        if self.eta_tilde <= 0:
            raise ValueError("eta - 0.5*gamma*tau must be positive")
        # Discrete-time Almgren-Chriss urgency parameter.
        arg=1.0 + (self.p.risk_aversion*self.p.sigma**2*self.tau**2)/(2*self.eta_tilde)
        return float(np.arccosh(arg)/self.tau)

    def inventory_trajectory(self) -> pd.Series:
        t=np.linspace(0,self.p.horizon,self.p.periods+1)
        k=self.kappa
        if abs(k)<1e-14:
            x=self.p.shares*(1-t/self.p.horizon)
        else:
            x=self.p.shares*np.sinh(k*(self.p.horizon-t))/np.sinh(k*self.p.horizon)
        return pd.Series(x,index=t,name="inventory")

    def trading_trajectory(self) -> pd.Series:
        x=self.inventory_trajectory().to_numpy()
        n=x[:-1]-x[1:]
        t=np.linspace(self.tau,self.p.horizon,self.p.periods)
        return pd.Series(n,index=t,name="shares_traded")

    def temporary_impact_cost(self) -> float:
        n=self.trading_trajectory().to_numpy()
        return float((self.p.eta/self.tau)*np.sum(n**2))

    def permanent_impact_cost(self) -> float:
        return float(0.5*self.p.gamma*self.p.shares**2)

    def spread_cost(self) -> float:
        n=self.trading_trajectory().to_numpy()
        return float(self.p.half_spread*np.sum(np.abs(n)))

    def expected_cost(self) -> float:
        return self.temporary_impact_cost()+self.permanent_impact_cost()+self.spread_cost()

    def cost_variance(self) -> float:
        x=self.inventory_trajectory().to_numpy()[:-1]
        return float(self.p.sigma**2*self.tau*np.sum(x**2))

    def objective(self) -> float:
        return self.expected_cost()+self.p.risk_aversion*self.cost_variance()

    def expected_shortfall_bps(self) -> float:
        return float(self.expected_cost()/(self.p.shares*self.p.price)*1e4)

    def schedule(self) -> pd.DataFrame:
        x=self.inventory_trajectory(); n=self.trading_trajectory()
        out=pd.DataFrame({"time":x.index,"inventory":x.values})
        out["trade_shares"]=0.0
        out.loc[1:,"trade_shares"]=n.values
        out["inventory_pct"]=out.inventory/self.p.shares
        out["trade_pct"]=out.trade_shares/self.p.shares
        return out

    def simulate_shortfall(self, n_sims=5000, seed=42) -> np.ndarray:
        rng=np.random.default_rng(seed)
        x=self.inventory_trajectory().to_numpy()[:-1]
        shocks=rng.normal(0,self.p.sigma*np.sqrt(self.tau),size=(n_sims,self.p.periods))
        stochastic=-np.sum(x*shocks,axis=1)
        return self.expected_cost()+stochastic

    def twap_expected_cost(self) -> float:
        n=np.repeat(self.p.shares/self.p.periods,self.p.periods)
        temporary=(self.p.eta/self.tau)*np.sum(n**2)
        permanent=0.5*self.p.gamma*self.p.shares**2
        spread=self.p.half_spread*np.sum(np.abs(n))
        return float(temporary+permanent+spread)

    def twap_cost_variance(self) -> float:
        x=self.p.shares*(1-np.arange(self.p.periods)/self.p.periods)
        return float(self.p.sigma**2*self.tau*np.sum(x**2))
