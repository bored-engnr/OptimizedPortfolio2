# Instalando demais biliotecas
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import pypfopt
import streamlit as st
import plotly.express as px

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import CLA, plotting
from pypfopt import DiscreteAllocation
from pypfopt import EfficientFrontier

tickers = ["IWDA.L", "EIMI.L", "EMVL.L", "USSC.L", "IWVL.L"]
values = [0.40, 0.15, 0.15, 0.15, 0.15]

ohlc = yf.download(tickers, period="max")

st.line_chart(ohlc['Adj Close'])

data = []
i = 0 
for ticker in tickers:
  stock = yf.Ticker(ticker)
  #print(stock.info['shortName'])
  data.append([ticker, stock.info['shortName'], values[i]])
  i += 1

# Criando Dataframe
df_carteira = pd.DataFrame(data, columns=['Ticker', 'Nome', '%'])

df_carteira

st.subheader("Martriz de Covariância")
prices = ohlc["Adj Close"].dropna(how="all")
sample_cov = risk_models.sample_cov(prices, frequency=252)
#sample_cov

#plotting.plot_covariance(sample_cov, plot_correlation=True)

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
plotting.plot_covariance(S, plot_correlation=True)

fig = px.imshow(S,text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# Estimativa de Retorno dos Ativos
st.subheader("Estimativa de Retorno dos Ativos")
mu = expected_returns.capm_return(prices)
mu.plot.barh(figsize=(10,6)).set_title('Retorno no Modelo CAPM')

st.bar_chart(mu)

# Max Sharpe com restrições por Setor
st.subheader("Max Sharpe")
ef = EfficientFrontier(mu, S)  # weight_bounds automatically set to (0, 1)
ef.max_sharpe()
weights = ef.clean_weights()
weights_series = list(weights.items())

df_sharpe_portfolio = pd.DataFrame(weights_series)
df_sharpe_portfolio = df_sharpe_portfolio.rename(columns={0: 'Stock', 1: 'Percentage'})
df_sharpe_portfolio

fig_pie_sharpe = px.pie(df_sharpe_portfolio[df_sharpe_portfolio['Percentage']>0], values='Percentage', names='Stock')
st.plotly_chart(fig_pie_sharpe, use_container_width=True)

st.subheader("Fronteira Eficiente")

cla = CLA(mu, S)
cla.max_sharpe()
ef_text = cla.portfolio_performance(verbose=True)
st.markdown(ef_text)

ax = plotting.plot_efficient_frontier(cla, showfig=False)
st.write(ax)