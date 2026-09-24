import streamlit as st
import numpy as np
from execution.model import ACParams, AlmgrenChriss
from execution.analysis import efficient_frontier

st.set_page_config(page_title="Almgren-Chriss",layout="wide")
st.title("Almgren-Chriss Optimal Execution Model")
shares=st.sidebar.number_input("Shares",1000,10000000,1000000,10000)
sigma=st.sidebar.slider("Volatility per sqrt(time)",0.001,0.10,0.02,0.001)
eta=st.sidebar.number_input("Temporary impact eta",value=2.5e-6,format="%.8f")
gamma=st.sidebar.number_input("Permanent impact gamma",value=2.5e-7,format="%.8f")
lam=st.sidebar.number_input("Risk aversion lambda",value=1e-6,format="%.8f")
p=ACParams(shares=shares,sigma=sigma,eta=eta,gamma=gamma,risk_aversion=lam)
m=AlmgrenChriss(p); sched=m.schedule()
c1,c2,c3=st.columns(3)
c1.metric("Expected cost","$"+format(m.expected_cost(),",.0f"))
c2.metric("Cost stdev","$"+format(np.sqrt(m.cost_variance()),",.0f"))
c3.metric("Kappa",format(m.kappa,".4f"))
st.subheader("Inventory trajectory"); st.line_chart(sched.set_index("time")[["inventory"]])
st.subheader("Trading trajectory"); st.bar_chart(sched.set_index("time")[["trade_shares"]])
st.subheader("Efficient frontier"); f=efficient_frontier(p); st.scatter_chart(f,x="stdev_cost",y="expected_cost")
