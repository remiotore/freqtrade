# Freqtrade

## 

### Analysis OHLCV+

[Analysis of OHLCV](analysis-ohlcv.ipynb) (Open, High, Low, Close, Volume and Indicators) data is performed to explore market behavior, price action, and volume trends. This section is a work in progress and will include various visualizations and statistical summaries of historical price data.

### Analysis of Strategies 

This repository includes an [automated analysis](analysis-strat_ninja.ipynb) of trading [strategies](strategies/) sourced from [Strat.Ninja](https://strat.ninja/). A large number of public strategies have been downloaded and processed (private strategies are excluded). The analysis extracts metadata and code features for each strategy, enabling further research and comparison.

The main analysis produces a dataframe (see `strategies_metadata.ndjson`) where each row represents a strategy and columns include:

- `strategy`: Name of the strategy
- `scope`: Public/Private
- `mode`: Spot or Futures
- `dca`: Whether DCA (Dollar Cost Averaging) is used
- `timeframe`: Main trading timeframe (e.g., 5m, 1h)
- `failed`: Failure status (if any)
- `bias`: Bias status (e.g., unbiased, lookahead bias)
- `stalled`: Stalled status (if any)
- `leverage`: Leverage used (if any)
- `short`: Whether shorting is supported
- `profit`: Average profit metric (from Strat.Ninja overview)
- `stoploss`: Stoploss value
- Indicator columns: Each indicator used by the strategy (e.g., `rsi`, `ema_50`, `macd`, etc.) is represented as a column with value 1 if present

This structure allows for correlation analysis, feature importance, and filtering strategies by their properties or indicators used.

---

## Thanks!

```
 ╔════════════════════════════════════════════════════════╗
 ║   If you enjoy my work, please consider donating ❤️    ║
 ║   Every small amount helps me to keep going, thanks!   ║
 ╠════════════════════════════════════════════════════════╣
 ║   BTC: 33C1LBYty9dx4H3ScShD91pAS6Gm15CZNB              ║
 ║   ETH: 0x06d46296a5eba0e2d9a2dffd9e3977fb3cc6030d      ║
 ║   SOL: gdsqfAEugaqzxpi5sq5wavA3TnJhok4tkFDeAnRovDa     ║
 ║   XRP: rLHzPsX6oXkzU2qL12kHCH8G8cnZv1rBJh              ║
 ╚════════════════════════════════════════════════════════╝
```