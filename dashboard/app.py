import streamlit as st
import numpy as np
from execution.model import ACParams, AlmgrenChriss
from execution.analysis import efficient_frontier, compare_to_twap

st.set_page_config(page_title="Almgren-Chriss",layout="wide")
st.title("Almgren-Chriss Optimal Execution & Transaction Cost Model")

shares=st.sidebar.number_input("Shares",1000,10000000,1000000,10000)
price=st.sidebar.number_input("Reference price",value=50.0,min_value=0.01)
sigma=st.sidebar.slider("Volatility per sqrt(time)",0.001,0.10,0.02,0.001)
eta=st.sidebar.number_input("Temporary impact eta",value=2.5e-6,format="%.8f")
gamma=st.sidebar.number_input("Permanent impact gamma",value=2.5e-7,format="%.8f")
lam=st.sidebar.number_input("Risk aversion lambda",value=1e-6,format="%.8f")
spread=st.sidebar.number_input("Half spread ($/share)",value=0.0,format="%.4f")

p=ACParams(shares=shares,price=price,sigma=sigma,eta=eta,gamma=gamma,risk_aversion=lam,half_spread=spread)
m=AlmgrenChriss(p)
sched=m.schedule()

c1,c2,c3,c4=st.columns(4)
c1.metric("Expected cost","$"+format(m.expected_cost(),",.0f"))
c2.metric("Expected cost",format(m.expected_shortfall_bps(),".2f")+" bps")
c3.metric("Cost stdev","$"+format(np.sqrt(m.cost_variance()),",.0f"))
c4.metric("Kappa",format(m.kappa,".4f"))

tabs=st.tabs(["Schedule","Cost Decomposition","Efficient Frontier","TWAP Comparison"])

with tabs[0]:
    st.subheader("Inventory trajectory")
    st.line_chart(sched.set_index("time")[["inventory"]])
    st.subheader("Trading trajectory")
    st.bar_chart(sched.set_index("time")[["trade_shares"]])

with tabs[1]:
    st.dataframe({
        "temporary_impact":[m.temporary_impact_cost()],
        "permanent_impact":[m.permanent_impact_cost()],
        "spread":[m.spread_cost()],
        "total":[m.expected_cost()]
    },use_container_width=True)

with tabs[2]:
    f=efficient_frontier(p)
    st.scatter_chart(f,x="stdev_cost",y="expected_cost")
    st.dataframe(f,use_container_width=True)

with tabs[3]:
    comp=compare_to_twap(p)
    st.dataframe(comp,use_container_width=True)
