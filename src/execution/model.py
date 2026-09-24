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
        self.p=params
        self.tau=params.horizon/params.periods

    @property
    def kappa(self) -> float:
        if self.p.risk_aversion <= 0 or self.p.sigma == 0:
            return 0.0
        return float(np.sqrt(self.p.risk_aversion*self.p.sigma**2/self.p.eta))

    def inventory_trajectory(self) -> pd.Series:
        t=np.linspace(0,self.p.horizon,self.p.periods+1)
        k=self.kappa
        if abs(k)<1e-14:
            x=self.p.shares*(1-t/self.p.horizon)
        else:
            x=self.p.shares*np.sinh(k*(self.p.horizon-t))/np.sinh(k*self.p.horizon)
        return pd.Series(x,index=t,name="inventory")

    def trading_trajectory(self) -> pd.Series:
        x=self.inventory_trajectory().values
        n=x[:-1]-x[1:]
        t=np.linspace(self.tau,self.p.horizon,self.p.periods)
        return pd.Series(n,index=t,name="shares_traded")

    def expected_cost(self) -> float:
        n=self.trading_trajectory().values
        temporary=(self.p.eta/self.tau)*np.sum(n**2)
        permanent=0.5*self.p.gamma*self.p.shares**2
        spread=self.p.half_spread*np.sum(np.abs(n))
        return float(temporary+permanent+spread)

    def cost_variance(self) -> float:
        x=self.inventory_trajectory().values[:-1]
        return float(self.p.sigma**2*self.tau*np.sum(x**2))

    def objective(self) -> float:
        return self.expected_cost()+self.p.risk_aversion*self.cost_variance()

    def schedule(self) -> pd.DataFrame:
        x=self.inventory_trajectory(); n=self.trading_trajectory()
        out=pd.DataFrame({"time":x.index,"inventory":x.values})
        out["trade_shares"]=0.0
        out.loc[1:,"trade_shares"]=n.values
        out["inventory_pct"]=out.inventory/self.p.shares
        return out

    def simulate_shortfall(self, n_sims=5000, seed=42) -> np.ndarray:
        rng=np.random.default_rng(seed)
        x=self.inventory_trajectory().values[:-1]
        shocks=rng.normal(0,self.p.sigma*np.sqrt(self.tau),size=(n_sims,self.p.periods))
        stochastic=-np.sum(x*shocks,axis=1)
        return self.expected_cost()+stochastic
