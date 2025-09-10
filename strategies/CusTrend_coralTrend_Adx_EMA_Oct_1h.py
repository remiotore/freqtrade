



from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union


from freqtrade.persistence import Trade

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, stoploss_from_open, DecimalParameter,
                                IntParameter, IStrategy, informative, merge_informative_pair)


import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib



'''
The "Coral Trend Indicator" is a trend-following indicator that uses a combination of exponential moving averages (EMAs) and the Commodity Channel Index (CCI) to identify trends and potential reversal points

This code imports the necessary libraries (pandas and numpy) and defines a function called coral_trend() that takes a DataFrame of cryptocurrency data (with a "close" column) and the EMA period, CCI period, and CCI threshold as inputs. The function first calculates the fast and slow exponential moving averages (EMAs) of the cryptocurrency and stores them in new columns called "ema_fast" and "ema_slow", respectively. It then calculates the Commodity Channel Index (CCI) of the cryptocurrency and stores it in a new column called "cci".

The function then determines the trend based on the EMAs and CCI, and stores it in a new column called "coral_trend".

The coral_trend() function determines the trend by using the np.where() function to set the value of the "coral_trend" column based on the following conditions:

    If the fast EMA is greater than the slow EMA and the CCI is less than the negative CCI threshold, then the trend is set to 1 (indicating an uptrend).
    If the fast EMA is less than the slow EMA and the CCI is greater than the positive CCI threshold, then the trend is set to -1 (indicating a downtrend).
    Otherwise, the trend is set to 0 (indicating no trend or a neutral market).

The coral_trend() function then returns the modified DataFrame with the "coral_trend" column added.
'''
def coral_trend(df, ema_period=10, cci_period=20, cci_threshold=100):

    df['ema_fast'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_period*2, adjust=False).mean()

    df['cci'] = ((df['close'] - df['close'].rolling(cci_period).mean()) / df['close'].rolling(cci_period).std()) * np.sqrt(cci_period)

    df['coral_trend'] = np.where((df['ema_fast'] > df['ema_slow']) & (df['cci'] < -cci_threshold), 1, 0)
    df['coral_trend'] = np.where((df['ema_fast'] < df['ema_slow']) & (df['cci'] > cci_threshold), -1, df['coral_trend'])
    
    return df


def determine_trend(df):
    df['trend'] = 0
    for i in range(2, len(df)):
        close_prev = df['close'].iloc[i-1]
        close = df['close'].iloc[i]
        high_prev = df['high'].iloc[i-1]
        low_prev = df['low'].iloc[i-1]
        open_prev = df['open'].iloc[i-1]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        open = df['open'].iloc[i] 

        if (
            close_prev < close and
            high_prev < high and
            high_prev < close and
            low_prev < low and  # Close of previous candle is lower than present candle
            close - open > (close_prev - open_prev) and

            open < close and 
            open_prev < close_prev and

            high_prev - close_prev < close_prev - open_prev and  # High - Close < Close - Open for previous candle
            high - close < close - open  # High - Close < Close - Open for present candle
        ):
            df.at[i, 'trend'] = 1

        elif (
            close_prev > close and
            high_prev > high and
            low_prev > low and   # Close of previous candle is higher than present candle
            low_prev > close and   # Close of previous candle is higher than present candle
            open > close and 
            open_prev > close_prev and

            open - close > (open_prev - close_prev) and

            close_prev - low_prev < open_prev - close_prev and  # Close - Low < Open - Close for previous candle
            close - low < open - close  # Close - Low < Open - Close for present candle
        ):
            df.at[i, 'trend'] = -1

    return df


class CusTrend_coralTrend_Adx_EMA_Oct_1h(IStrategy):


    INTERFACE_VERSION = 3

    timeframe = '1h'

    can_short = True



    '''

 




sudo docker-compose run freqtrade backtesting --strategy CusTrend_coralTrend_Adx_EMA_Oct_1h -i 1h --export trades --breakdown month --timerange 20210101-20230815
2023-10-03 15:12:26,838 - freqtrade - INFO - freqtrade 2023.7
2023-10-03 15:12:34,359 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'stake_amount' with value in config file: 200.
2023-10-03 15:12:34,359 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'unfilledtimeout' with value in config file: {'entry': 10, 'exit': 10, 'exit_timeout_count': 0, 'unit': 'minutes'}.
2023-10-03 15:12:34,359 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'max_open_trades' with value in config file: 4.
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using minimal_roi: {'0': 0.101, '373': 0.068, '1088': 0.025, '1336': 0}
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using timeframe: 1h
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stoploss: -0.347
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop: True
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive: 0.01
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive_offset: 0.012
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_only_offset_is_reached: True
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_custom_stoploss: False
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using process_only_new_candles: True
2023-10-03 15:12:34,360 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_types: {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
2023-10-03 15:12:34,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}
2023-10-03 15:12:34,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_currency: USDT
2023-10-03 15:12:34,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_amount: 200
2023-10-03 15:12:34,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using protections: [{'method': 'MaxDrawdown', 'lookback_period_candles': 2, 'trade_limit': 1, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.1}, {'method': 'StoplossGuard', 'lookback_period_candles': 8, 'trade_limit': 1, 'stop_duration_candles': 4, 'only_per_pair': False}]
2023-10-03 15:12:34,361 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using startup_candle_count: 30


2023-10-03 15:14:52,156 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy CusTrend_coralTrend_Adx_EMA_Oct_1h
2023-10-03 15:14:52,156 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_long_max = 50
2023-10-03 15:14:52,157 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_long_min = 21
2023-10-03 15:14:52,157 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_short_max = 51
2023-10-03 15:14:52,157 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_short_min = 13
2023-10-03 15:14:52,157 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ema_period = 142
2023-10-03 15:14:52,157 - freqtrade.strategy.hyper - INFO - Strategy Parameter: leverage_num = 4
2023-10-03 15:14:52,158 - freqtrade.strategy.hyper - INFO - Strategy Parameter: volume_check = 22
2023-10-03 15:14:52,158 - freqtrade.strategy.hyper - INFO - Strategy Parameter: sell_shift = 5
2023-10-03 15:14:52,158 - freqtrade.strategy.hyper - INFO - Strategy Parameter: sell_shift_short = 5
2023-10-03 15:14:52,158 - freqtrade.strategy.hyper - INFO - Strategy Parameter: volume_check_exit = 19
2023-10-03 15:14:52,159 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_allowed_drawdown = 0.1
2023-10-03 15:14:52,159 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_drawdown_lookback = 2
2023-10-03 15:14:52,159 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_drawdown_stop_duration = 4
2023-10-03 15:14:52,159 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_drawdown_trade_limit = 1
2023-10-03 15:14:52,159 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stoploss_guard_lookback = 8
2023-10-03 15:14:52,159 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stoploss_guard_stop_duration = 4
2023-10-03 15:14:52,160 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stoploss_guard_trade_limit = 1


=========================================================== ENTER TAG STATS ===========================================================
|   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|-------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
| TOTAL |     19512 |           1.53 |       29904.97 |         59666.204 |        5966.62 |        1:27:00 | 15205     0  4307  77.9 |
======================================================= EXIT REASON STATS ========================================================
|        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
| trailing_stop_loss |   13767 |  11305     0  2462  82.1 |           1.5  |       20712.3  |         41334.9   |        5178.08 |
|                roi |    3912 |   3899     0    13  99.7 |           9.93 |       38849.4  |         77530.3   |        9712.34 |
|          exit_long |     846 |      1     0   845   0.1 |         -14.05 |      -11885.5  |        -23720.7   |       -2971.38 |
|         exit_short |     723 |      0     0   723     0 |         -11.59 |       -8378.26 |        -16724.4   |       -2094.57 |
|          stop_loss |     259 |      0     0   259     0 |         -34.87 |       -9030.74 |        -18029.8   |       -2257.69 |
|        liquidation |       4 |      0     0     4     0 |         -89.06 |        -356.25 |          -712.153 |         -89.06 |
|         force_exit |       1 |      0     0     1     0 |          -5.95 |          -5.95 |           -11.907 |          -1.49 |
======================= MONTH BREAKDOWN ========================
|      Month |   Tot Profit USDT |   Wins |   Draws |   Losses |
|------------+-------------------+--------+---------+----------|
| 31/01/2021 |          2053.09  |    377 |       0 |       93 |
| 28/02/2021 |          2911.42  |    532 |       0 |       93 |
| 31/03/2021 |          3454.87  |    619 |       0 |      153 |
| 30/04/2021 |          3404.27  |    597 |       0 |      135 |
| 31/05/2021 |          2013.48  |    531 |       0 |      120 |
| 30/06/2021 |          2163.77  |    479 |       0 |      114 |
| 31/07/2021 |          1982.28  |    543 |       0 |      135 |
| 31/08/2021 |          1980.32  |    557 |       0 |      155 |
| 30/09/2021 |          3063.71  |    483 |       0 |      107 |
| 31/10/2021 |          2045.31  |    540 |       0 |      162 |
| 30/11/2021 |          3190.49  |    629 |       0 |      152 |
| 31/12/2021 |          2820.53  |    540 |       0 |      123 |
| 31/01/2022 |          2365.55  |    504 |       0 |      116 |
| 28/02/2022 |          1496.26  |    449 |       0 |      145 |
| 31/03/2022 |          1702.11  |    521 |       0 |      168 |
| 30/04/2022 |          2633.78  |    522 |       0 |      147 |
| 31/05/2022 |          2176.6   |    514 |       0 |      120 |
| 30/06/2022 |          2342.53  |    495 |       0 |      138 |
| 31/07/2022 |          2140.83  |    483 |       0 |      152 |
| 31/08/2022 |          1366.66  |    429 |       0 |      146 |
| 30/09/2022 |           956.942 |    417 |       0 |      148 |
| 31/10/2022 |          1152.5   |    405 |       0 |      146 |
| 30/11/2022 |          1286.25  |    424 |       0 |      126 |
| 31/12/2022 |           883.354 |    418 |       0 |      155 |
| 31/01/2023 |          1521.18  |    462 |       0 |      138 |
| 28/02/2023 |          2311.52  |    491 |       0 |      113 |
| 31/03/2023 |           664.388 |    428 |       0 |      154 |
| 30/04/2023 |          1015.03  |    413 |       0 |      129 |
| 31/05/2023 |           677.527 |    389 |       0 |      134 |
| 30/06/2023 |          1286.08  |    453 |       0 |      133 |
| 31/07/2023 |           478.821 |    368 |       0 |      177 |
| 31/08/2023 |           124.745 |    193 |       0 |       80 |
=================== SUMMARY METRICS ====================
| Metric                      | Value                  |
|-----------------------------+------------------------|
| Backtesting from            | 2021-01-02 06:00:00    |
| Backtesting to              | 2023-08-15 00:00:00    |
| Max open trades             | 4                      |
|                             |                        |
| Total/Daily Avg Trades      | 19512 / 20.45          |
| Starting balance            | 1000 USDT              |
| Final balance               | 60666.204 USDT         |
| Absolute profit             | 59666.204 USDT         |
| Total profit %              | 5966.62%               |
| CAGR %                      | 381.01%                |
| Sortino                     | 63.30                  |
| Sharpe                      | 74.94                  |
| Calmar                      | 8256.74                |
| Profit factor               | 1.72                   |
| Expectancy (Ratio)          | 3.06 (0.16)            |
| Trades per day              | 20.45                  |
| Avg. daily profit %         | 6.25%                  |
| Avg. stake amount           | 199.529 USDT           |
| Total trade volume          | 3893209.107 USDT       |
|                             |                        |
| Long / Short                | 11155 / 8357           |
| Total profit Long %         | 3511.16%               |
| Total profit Short %        | 2455.46%               |
| Absolute profit Long        | 35111.567 USDT         |
| Absolute profit Short       | 24554.637 USDT         |
|                             |                        |
| Best Pair                   | CRV/USDT:USDT 1223.38% |
| Worst Pair                  | CHZ/USDT:USDT -64.24%  |
| Best trade                  | BEL/USDT:USDT 13.43%   |
| Worst trade                 | FTM/USDT:USDT -92.36%  |
| Best day                    | 416.069 USDT           |
| Worst day                   | -295.425 USDT          |
| Days win/draw/lose          | 718 / 0 / 231          |
| Avg. Duration Winners       | 0:53:00                |
| Avg. Duration Loser         | 3:28:00                |
| Max Consecutive Wins / Loss | 41 / 9                 |
| Rejected Entry signals      | 34147                  |
| Entry/Exit Timeouts         | 0 / 5491               |
|                             |                        |
| Min balance                 | 1020.199 USDT          |
| Max balance                 | 60788.167 USDT         |
| Max % of account underwater | 22.01%                 |
| Absolute Drawdown (Account) | 1.45%                  |
| Absolute Drawdown           | 597.565 USDT           |
| Drawdown high               | 40292.12 USDT          |
| Drawdown low                | 39694.556 USDT         |
| Drawdown Start              | 2022-05-11 12:00:00    |
| Drawdown End                | 2022-05-11 13:00:00    |
| Market change               | 23.42%                 |
========================================================

Backtested 2021-01-02 06:00:00 -> 2023-08-15 00:00:00 | Max open trades : 4
==================================================================================== STRATEGY SUMMARY ====================================================================================
|                           Strategy |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |            Drawdown |
|------------------------------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------+---------------------|
| CusTrend_coralTrend_Adx_EMA_Oct_1h |     19512 |           1.53 |       29904.97 |         59666.204 |        5966.62 |        1:27:00 | 15205     0  4307  77.9 | 597.565 USDT  1.45% |
==========================================================================================================================================================================================






(((((((below results are with protections enabled during backtest))))))


sudo docker-compose run freqtrade backtesting --strategy CusTrend_coralTrend_Adx_EMA_Oct_1h -i 1h --export trades --breakdown month --timerange 20210101-20230815 --enable-protections

2023-10-06 14:12:49,859 - freqtrade.strategy.hyper - INFO - Loading parameters from file /freqtrade/user_data/strategies/CusTrend_coralTrend_Adx_EMA_Oct_1h.json
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'timeframe' with value in config file: 1h.
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'stake_currency' with value in config file: USDT.
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'stake_amount' with value in config file: 200.
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'unfilledtimeout' with value in config file: {'entry': 10, 'exit': 10, 'exit_timeout_count': 0, 'unit': 'minutes'}.
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'max_open_trades' with value in config file: 4.
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using minimal_roi: {'0': 0.101, '373': 0.068, '1088': 0.025, '1336': 0}
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using timeframe: 1h
2023-10-06 14:12:49,860 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stoploss: -0.347
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop: True
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive: 0.01
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive_offset: 0.012
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_only_offset_is_reached: True
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_custom_stoploss: False
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using process_only_new_candles: True
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_types: {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_currency: USDT
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_amount: 200
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using protections: [{'method': 'MaxDrawdown', 'lookback_period_candles': 2, 'trade_limit': 1, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.1}, {'method': 'StoplossGuard', 'lookback_period_candles': 8, 'trade_limit': 1, 'stop_duration_candles': 4, 'only_per_pair': False}]
2023-10-06 14:12:49,861 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using startup_candle_count: 30
2023-10-06 14:12:49,862 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using unfilledtimeout: {'entry': 10, 'exit': 10, 'exit_timeout_count': 0, 'unit': 'minutes'}

2023-10-06 14:15:06,723 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy CusTrend_coralTrend_Adx_EMA_Oct_1h
2023-10-06 14:15:06,724 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_long_max = 50
2023-10-06 14:15:06,724 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_long_min = 21
2023-10-06 14:15:06,724 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_short_max = 51
2023-10-06 14:15:06,724 - freqtrade.strategy.hyper - INFO - Strategy Parameter: adx_short_min = 13
2023-10-06 14:15:06,724 - freqtrade.strategy.hyper - INFO - Strategy Parameter: ema_period = 142
2023-10-06 14:15:06,724 - freqtrade.strategy.hyper - INFO - Strategy Parameter: leverage_num = 4
2023-10-06 14:15:06,725 - freqtrade.strategy.hyper - INFO - Strategy Parameter: volume_check = 22
2023-10-06 14:15:06,725 - freqtrade.strategy.hyper - INFO - Strategy Parameter: sell_shift = 5
2023-10-06 14:15:06,726 - freqtrade.strategy.hyper - INFO - Strategy Parameter: sell_shift_short = 5
2023-10-06 14:15:06,726 - freqtrade.strategy.hyper - INFO - Strategy Parameter: volume_check_exit = 19
2023-10-06 14:15:06,726 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_allowed_drawdown = 0.1
2023-10-06 14:15:06,726 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_drawdown_lookback = 2
2023-10-06 14:15:06,727 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_drawdown_stop_duration = 4
2023-10-06 14:15:06,727 - freqtrade.strategy.hyper - INFO - Strategy Parameter: max_drawdown_trade_limit = 1
2023-10-06 14:15:06,727 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stoploss_guard_lookback = 8
2023-10-06 14:15:06,727 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stoploss_guard_stop_duration = 4
2023-10-06 14:15:06,727 - freqtrade.strategy.hyper - INFO - Strategy Parameter: stoploss_guard_trade_limit = 1


========================================================== LEFT OPEN TRADES REPORT ============================================================
|           Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|----------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
| GALA/USDT:USDT |         1 |          -5.95 |          -5.95 |           -11.907 |          -1.19 |        7:00:00 |     0     0     1     0 |
|          TOTAL |         1 |          -5.95 |          -5.95 |           -11.907 |          -1.19 |        7:00:00 |     0     0     1     0 |
=========================================================== ENTER TAG STATS ===========================================================
|   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|-------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
| TOTAL |     12791 |           1.56 |       19950.24 |         39782.063 |        3978.21 |        1:30:00 |  9959     0  2832  77.9 |
======================================================= EXIT REASON STATS ========================================================
|        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
| trailing_stop_loss |    8977 |   7408     0  1569  82.5 |           1.53 |       13742.8  |         27415.6   |        3435.7  |
|                roi |    2561 |   2550     0    11  99.6 |           9.94 |       25457.8  |         50795.8   |        6364.46 |
|          exit_long |     557 |      1     0   556   0.2 |         -13.54 |       -7541.39 |        -15057.9   |       -1885.35 |
|         exit_short |     537 |      0     0   537     0 |         -11.43 |       -6136.11 |        -12252.9   |       -1534.03 |
|          stop_loss |     157 |      0     0   157     0 |         -34.88 |       -5476.45 |        -10929.7   |       -1369.11 |
|        liquidation |       1 |      0     0     1     0 |         -90.49 |         -90.49 |          -176.769 |         -22.62 |
|         force_exit |       1 |      0     0     1     0 |          -5.95 |          -5.95 |           -11.907 |          -1.49 |
======================= MONTH BREAKDOWN ========================
|      Month |   Tot Profit USDT |   Wins |   Draws |   Losses |
|------------+-------------------+--------+---------+----------|
| 31/01/2021 |          1275.61  |    240 |       0 |       54 |
| 28/02/2021 |          2852.33  |    396 |       0 |       60 |
| 31/03/2021 |          2069.72  |    369 |       0 |       91 |
| 30/04/2021 |          2386.87  |    368 |       0 |       79 |
| 31/05/2021 |          1600.44  |    353 |       0 |       69 |
| 30/06/2021 |          1250.87  |    307 |       0 |       82 |
| 31/07/2021 |          1149.51  |    332 |       0 |       95 |
| 31/08/2021 |           927.824 |    321 |       0 |      108 |
| 30/09/2021 |          1609     |    301 |       0 |       71 |
| 31/10/2021 |          1141.03  |    320 |       0 |      104 |
| 30/11/2021 |          1895.8   |    357 |       0 |       83 |
| 31/12/2021 |          1609.43  |    365 |       0 |       86 |
| 31/01/2022 |          1689.36  |    349 |       0 |       77 |
| 28/02/2022 |          1171.06  |    276 |       0 |       78 |
| 31/03/2022 |          1243.72  |    344 |       0 |       97 |
| 30/04/2022 |          1821.71  |    350 |       0 |       91 |
| 31/05/2022 |          2133.85  |    370 |       0 |       74 |
| 30/06/2022 |          1874.28  |    347 |       0 |       88 |
| 31/07/2022 |          1003.94  |    277 |       0 |       96 |
| 31/08/2022 |            98.924 |    262 |       0 |      105 |
| 30/09/2022 |           720.696 |    284 |       0 |      102 |
| 31/10/2022 |           376.107 |    260 |       0 |      108 |
| 30/11/2022 |           833.128 |    286 |       0 |       90 |
| 31/12/2022 |           780.569 |    295 |       0 |      107 |
| 31/01/2023 |          1443.17  |    308 |       0 |       95 |
| 28/02/2023 |          1562.92  |    319 |       0 |       78 |
| 31/03/2023 |           796.425 |    320 |       0 |      107 |
| 30/04/2023 |           651.015 |    291 |       0 |       96 |
| 31/05/2023 |           443.557 |    281 |       0 |      103 |
| 30/06/2023 |           634.855 |    313 |       0 |       97 |
| 31/07/2023 |           528.381 |    241 |       0 |      104 |
| 31/08/2023 |           205.956 |    157 |       0 |       57 |
==================== SUMMARY METRICS ====================
| Metric                      | Value                   |
|-----------------------------+-------------------------|
| Backtesting from            | 2021-01-02 06:00:00     |
| Backtesting to              | 2023-08-15 00:00:00     |
| Max open trades             | 4                       |
|                             |                         |
| Total/Daily Avg Trades      | 12791 / 13.41           |
| Starting balance            | 1000 USDT               |
| Final balance               | 40782.063 USDT          |
| Absolute profit             | 39782.063 USDT          |
| Total profit %              | 3978.21%                |
| CAGR %                      | 313.20%                 |
| Sortino                     | 43.84                   |
| Sharpe                      | 51.00                   |
| Calmar                      | 2220.01                 |
| Profit factor               | 1.75                    |
| Expectancy (Ratio)          | 3.11 (0.17)             |
| Trades per day              | 13.41                   |
| Avg. daily profit %         | 4.17%                   |
| Avg. stake amount           | 199.546 USDT            |
| Total trade volume          | 2552392.418 USDT        |
|                             |                         |
| Long / Short                | 7084 / 5707             |
| Total profit Long %         | 2402.00%                |
| Total profit Short %        | 1576.21%                |
| Absolute profit Long        | 24019.99 USDT           |
| Absolute profit Short       | 15762.073 USDT          |
|                             |                         |
| Best Pair                   | AAVE/USDT:USDT 590.51%  |
| Worst Pair                  | ARPA/USDT:USDT -118.23% |
| Best trade                  | BEL/USDT:USDT 13.43%    |
| Worst trade                 | KSM/USDT:USDT -90.49%   |
| Best day                    | 361.297 USDT            |
| Worst day                   | -255.427 USDT           |
| Days win/draw/lose          | 665 / 0 / 284           |
| Avg. Duration Winners       | 0:53:00                 |
| Avg. Duration Loser         | 3:39:00                 |
| Max Consecutive Wins / Loss | 39 / 7                  |
| Rejected Entry signals      | 21224                   |
| Entry/Exit Timeouts         | 0 / 3623                |
|                             |                         |
| Min balance                 | 1020.199 USDT           |
| Max balance                 | 40793.97 USDT           |
| Max % of account underwater | 15.36%                  |
| Absolute Drawdown (Account) | 3.59%                   |
| Absolute Drawdown           | 453.002 USDT            |
| Drawdown high               | 11623.195 USDT          |
| Drawdown low                | 11170.192 USDT          |
| Drawdown Start              | 2021-06-30 02:00:00     |
| Drawdown End                | 2021-07-05 18:00:00     |
| Market change               | 26.08%                  |
=========================================================

Backtested 2021-01-02 06:00:00 -> 2023-08-15 00:00:00 | Max open trades : 4
==================================================================================== STRATEGY SUMMARY ====================================================================================
|                           Strategy |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |            Drawdown |
|------------------------------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------+---------------------|
| CusTrend_coralTrend_Adx_EMA_Oct_1h |     12791 |           1.56 |       19950.24 |         39782.063 |        3978.21 |        1:30:00 |  9959     0  2832  77.9 | 453.002 USDT  3.59% |
==========================================================================================================================================================================================





(((((Below is with leverage of 3, remaining all same)))))


sudo docker-compose run freqtrade backtesting --strategy CusTrend_coralTrend_Adx_EMA_Oct_1h -i 1h --export trades --breakdown month --timerange 20210101-20230815

2023-10-06 15:06:59,305 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'unfilledtimeout' with value in config file: {'entry': 10, 'exit': 10, 'exit_timeout_count': 0, 'unit': 'minutes'}.
2023-10-06 15:06:59,305 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'max_open_trades' with value in config file: 4.
2023-10-06 15:06:59,305 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using minimal_roi: {'0': 0.101, '373': 0.068, '1088': 0.025, '1336': 0}
2023-10-06 15:06:59,305 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using timeframe: 1h
2023-10-06 15:06:59,305 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stoploss: -0.347
2023-10-06 15:06:59,305 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop: True
2023-10-06 15:06:59,305 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive: 0.01
2023-10-06 15:06:59,306 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive_offset: 0.012
2023-10-06 15:06:59,306 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_only_offset_is_reached: True
2023-10-06 15:06:59,306 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_custom_stoploss: False
2023-10-06 15:06:59,306 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using process_only_new_candles: True
2023-10-06 15:06:59,306 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_types: {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
2023-10-06 15:06:59,306 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}
2023-10-06 15:06:59,307 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_currency: USDT
2023-10-06 15:06:59,307 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_amount: 200
2023-10-06 15:06:59,307 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using protections: [{'method': 'MaxDrawdown', 'lookback_period_candles': 2, 'trade_limit': 1, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.1}, {'method': 'StoplossGuard', 'lookback_period_candles': 8, 'trade_limit': 1, 'stop_duration_candles': 4, 'only_per_pair': False}]

2023-10-06 15:09:19,196 - freqtrade.strategy.hyper - INFO - No params for buy found, using default values.
2023-10-06 15:09:19,197 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_long_max = 50
2023-10-06 15:09:19,197 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_long_min = 21
2023-10-06 15:09:19,197 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_short_max = 51
2023-10-06 15:09:19,197 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_short_min = 13
2023-10-06 15:09:19,198 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ema_period = 142
2023-10-06 15:09:19,198 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): leverage_num = 3
2023-10-06 15:09:19,198 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): volume_check = 22
2023-10-06 15:09:19,198 - freqtrade.strategy.hyper - INFO - No params for sell found, using default values.
2023-10-06 15:09:19,199 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_shift = 5
2023-10-06 15:09:19,199 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_shift_short = 5
2023-10-06 15:09:19,199 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): volume_check_exit = 19
2023-10-06 15:09:19,199 - freqtrade.strategy.hyper - INFO - No params for protection found, using default values.
2023-10-06 15:09:19,200 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_allowed_drawdown = 0.1
2023-10-06 15:09:19,200 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_drawdown_lookback = 2
2023-10-06 15:09:19,200 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_drawdown_stop_duration = 4
2023-10-06 15:09:19,200 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_drawdown_trade_limit = 1
2023-10-06 15:09:19,200 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stoploss_guard_lookback = 8
2023-10-06 15:09:19,200 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stoploss_guard_stop_duration = 4
2023-10-06 15:09:19,201 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stoploss_guard_trade_limit = 1

=========================================================== ENTER TAG STATS ===========================================================
|   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|-------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
| TOTAL |     18751 |           1.15 |       21622.33 |         43118.159 |        4311.82 |        1:44:00 | 15103     0  3648  80.5 |
======================================================= EXIT REASON STATS ========================================================
|        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
| trailing_stop_loss |   14214 |  12677     0  1537  89.2 |           1.58 |       22404.4  |         44669.5   |        5601.09 |
|                roi |    2449 |   2425     0    24  99.0 |           9.8  |       24004.2  |         47867     |        6001.05 |
|          exit_long |    1073 |      1     0  1072   0.1 |         -11.68 |      -12528.3  |        -24970.7   |       -3132.07 |
|         exit_short |     907 |      0     0   907     0 |          -9.33 |       -8460.04 |        -16883.5   |       -2115.01 |
|          stop_loss |     106 |      0     0   106     0 |         -34.91 |       -3700.76 |         -7369.9   |        -925.19 |
|        liquidation |       1 |      0     0     1     0 |         -92.7  |         -92.7  |          -185.369 |         -23.18 |
|         force_exit |       1 |      0     0     1     0 |          -4.46 |          -4.46 |            -8.93  |          -1.12 |
======================= MONTH BREAKDOWN ========================
|      Month |   Tot Profit USDT |   Wins |   Draws |   Losses |
|------------+-------------------+--------+---------+----------|
| 31/01/2021 |          1756.89  |    380 |       0 |       76 |
| 28/02/2021 |          2441.77  |    541 |       0 |       83 |
| 31/03/2021 |          2451.61  |    622 |       0 |      127 |
| 30/04/2021 |          2758.34  |    610 |       0 |      103 |
| 31/05/2021 |          1283.99  |    527 |       0 |      105 |
| 30/06/2021 |          1222.95  |    493 |       0 |       87 |
| 31/07/2021 |          1916.85  |    560 |       0 |      108 |
| 31/08/2021 |          1450.2   |    552 |       0 |      128 |
| 30/09/2021 |          2207.33  |    481 |       0 |       81 |
| 31/10/2021 |          1117.35  |    504 |       0 |      136 |
| 30/11/2021 |          2696.59  |    629 |       0 |      119 |
| 31/12/2021 |          1585.04  |    536 |       0 |      111 |
| 31/01/2022 |          1230.31  |    494 |       0 |      112 |
| 28/02/2022 |          1145.33  |    467 |       0 |      114 |
| 31/03/2022 |          1360.21  |    526 |       0 |      134 |
| 30/04/2022 |          1865.96  |    516 |       0 |      130 |
| 31/05/2022 |          2285.81  |    534 |       0 |       92 |
| 30/06/2022 |          1999.68  |    510 |       0 |      115 |
| 31/07/2022 |          1214.64  |    466 |       0 |      136 |
| 31/08/2022 |           932.961 |    440 |       0 |      115 |
| 30/09/2022 |           640.456 |    419 |       0 |      121 |
| 31/10/2022 |           825.297 |    381 |       0 |      125 |
| 30/11/2022 |          1214.93  |    426 |       0 |      104 |
| 31/12/2022 |           656.156 |    401 |       0 |      136 |
| 31/01/2023 |          1114.01  |    448 |       0 |      121 |
| 28/02/2023 |          1468.54  |    471 |       0 |      104 |
| 31/03/2023 |           422.165 |    433 |       0 |      135 |
| 30/04/2023 |           477.42  |    389 |       0 |      119 |
| 31/05/2023 |           337.294 |    367 |       0 |      121 |
| 30/06/2023 |           720.648 |    421 |       0 |      122 |
| 31/07/2023 |            90.746 |    358 |       0 |      152 |
| 31/08/2023 |           226.676 |    201 |       0 |       76 |
=================== SUMMARY METRICS ====================
| Metric                      | Value                  |
|-----------------------------+------------------------|
| Backtesting from            | 2021-01-02 06:00:00    |
| Backtesting to              | 2023-08-15 00:00:00    |
| Max open trades             | 4                      |
|                             |                        |
| Total/Daily Avg Trades      | 18751 / 19.66          |
| Starting balance            | 1000 USDT              |
| Final balance               | 44118.159 USDT         |
| Absolute profit             | 43118.159 USDT         |
| Total profit %              | 4311.82%               |
| CAGR %                      | 325.82%                |
| Sortino                     | 55.87                  |
| Sharpe                      | 67.65                  |
| Calmar                      | 2030.83                |
| Profit factor               | 1.71                   |
| Expectancy (Ratio)          | 2.30 (0.14)            |
| Trades per day              | 19.66                  |
| Avg. daily profit %         | 4.52%                  |
| Avg. stake amount           | 199.354 USDT           |
| Total trade volume          | 3738082.373 USDT       |
|                             |                        |
| Long / Short                | 10717 / 8034           |
| Total profit Long %         | 2671.97%               |
| Total profit Short %        | 1639.84%               |
| Absolute profit Long        | 26719.715 USDT         |
| Absolute profit Short       | 16398.443 USDT         |
|                             |                        |
| Best Pair                   | NEAR/USDT:USDT 641.81% |
| Worst Pair                  | CHZ/USDT:USDT -100.65% |
| Best trade                  | BEL/USDT:USDT 12.60%   |
| Worst trade                 | DOT/USDT:USDT -92.70%  |
| Best day                    | 394.869 USDT           |
| Worst day                   | -242.901 USDT          |
| Days win/draw/lose          | 687 / 0 / 262          |
| Avg. Duration Winners       | 1:00:00                |
| Avg. Duration Loser         | 4:47:00                |
| Max Consecutive Wins / Loss | 69 / 8                 |
| Rejected Entry signals      | 34969                  |
| Entry/Exit Timeouts         | 0 / 3632               |
|                             |                        |
| Min balance                 | 1020.199 USDT          |
| Max balance                 | 44143.254 USDT         |
| Max % of account underwater | 11.94%                 |
| Absolute Drawdown (Account) | 4.25%                  |
| Absolute Drawdown           | 463.643 USDT           |
| Drawdown high               | 9904.341 USDT          |
| Drawdown low                | 9440.698 USDT          |
| Drawdown Start              | 2021-05-07 19:00:00    |
| Drawdown End                | 2021-05-10 03:00:00    |
| Market change               | 26.08%                 |
========================================================

Backtested 2021-01-02 06:00:00 -> 2023-08-15 00:00:00 | Max open trades : 4
==================================================================================== STRATEGY SUMMARY ====================================================================================
|                           Strategy |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |            Drawdown |
|------------------------------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------+---------------------|
| CusTrend_coralTrend_Adx_EMA_Oct_1h |     18751 |           1.15 |       21622.33 |         43118.159 |        4311.82 |        1:44:00 | 15103     0  3648  80.5 | 463.643 USDT  4.25% |
==========================================================================================================================================================================================









((((((((((below is with leverage 3 and protections on while doing backtest))))))))))



sudo docker-compose run freqtrade backtesting --strategy CusTrend_coralTrend_Adx_EMA_Oct_1h -i 1h --export trades --breakdown month --timerange 20210101-20230815 --enable-protections


2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'timeframe' with value in config file: 1h.
2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'stake_currency' with value in config file: USDT.
2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'stake_amount' with value in config file: 200.
2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'unfilledtimeout' with value in config file: {'entry': 10, 'exit': 10, 'exit_timeout_count': 0, 'unit': 'minutes'}.
2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Override strategy 'max_open_trades' with value in config file: 4.
2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using minimal_roi: {'0': 0.101, '373': 0.068, '1088': 0.025, '1336': 0}
2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using timeframe: 1h
2023-10-06 15:53:55,318 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stoploss: -0.347
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop: True
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive: 0.01
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_stop_positive_offset: 0.012
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using trailing_only_offset_is_reached: True
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using use_custom_stoploss: False
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using process_only_new_candles: True
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_types: {'entry': 'limit', 'exit': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using order_time_in_force: {'entry': 'GTC', 'exit': 'GTC'}
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_currency: USDT
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using stake_amount: 200
2023-10-06 15:53:55,319 - freqtrade.resolvers.strategy_resolver - INFO - Strategy using protections: [{'method': 'MaxDrawdown', 'lookback_period_candles': 2, 'trade_limit': 1, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.1}, {'method': 'StoplossGuard', 'lookback_period_candles': 8, 'trade_limit': 1, 'stop_duration_candles': 4, 'only_per_pair': False}]

2023-10-06 15:56:17,698 - freqtrade.optimize.backtesting - INFO - Running backtesting for Strategy CusTrend_coralTrend_Adx_EMA_Oct_1h
2023-10-06 15:56:17,699 - freqtrade.strategy.hyper - INFO - No params for buy found, using default values.
2023-10-06 15:56:17,699 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_long_max = 50
2023-10-06 15:56:17,699 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_long_min = 21
2023-10-06 15:56:17,700 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_short_max = 51
2023-10-06 15:56:17,700 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): adx_short_min = 13
2023-10-06 15:56:17,700 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): ema_period = 142
2023-10-06 15:56:17,700 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): leverage_num = 3
2023-10-06 15:56:17,701 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): volume_check = 22
2023-10-06 15:56:17,701 - freqtrade.strategy.hyper - INFO - No params for sell found, using default values.
2023-10-06 15:56:17,702 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_shift = 5
2023-10-06 15:56:17,702 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): sell_shift_short = 5
2023-10-06 15:56:17,702 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): volume_check_exit = 19
2023-10-06 15:56:17,702 - freqtrade.strategy.hyper - INFO - No params for protection found, using default values.
2023-10-06 15:56:17,703 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_allowed_drawdown = 0.1
2023-10-06 15:56:17,703 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_drawdown_lookback = 2
2023-10-06 15:56:17,703 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_drawdown_stop_duration = 4
2023-10-06 15:56:17,703 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): max_drawdown_trade_limit = 1
2023-10-06 15:56:17,703 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stoploss_guard_lookback = 8
2023-10-06 15:56:17,703 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stoploss_guard_stop_duration = 4
2023-10-06 15:56:17,703 - freqtrade.strategy.hyper - INFO - Strategy Parameter(default): stoploss_guard_trade_limit = 1

=========================================================== ENTER TAG STATS ===========================================================
|   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|-------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
| TOTAL |     13787 |           1.14 |       15659.12 |         31249.212 |        3124.92 |        1:46:00 | 11114     0  2673  80.6 |
======================================================= EXIT REASON STATS ========================================================
|        Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|--------------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
| trailing_stop_loss |   10421 |   9381     0  1040  90.0 |           1.62 |       16930.7  |         33776.4   |        4232.68 |
|                roi |    1753 |   1732     0    21  98.8 |           9.8  |       17173.8  |         34270.3   |        4293.46 |
|          exit_long |     823 |      1     0   822   0.1 |         -11.3  |       -9297.24 |        -18547.3   |       -2324.31 |
|         exit_short |     718 |      0     0   718     0 |          -9.33 |       -6699.65 |        -13378.4   |       -1674.91 |
|          stop_loss |      70 |      0     0    70     0 |         -34.9  |       -2442.75 |         -4860.1   |        -610.69 |
|         force_exit |       2 |      0     0     2     0 |          -2.91 |          -5.83 |           -11.659 |          -1.46 |
======================= MONTH BREAKDOWN ========================
|      Month |   Tot Profit USDT |   Wins |   Draws |   Losses |
|------------+-------------------+--------+---------+----------|
| 31/01/2021 |          1160.83  |    258 |       0 |       50 |
| 28/02/2021 |          2190.39  |    417 |       0 |       57 |
| 31/03/2021 |          1464.35  |    391 |       0 |       87 |
| 30/04/2021 |          2040.94  |    437 |       0 |       67 |
| 31/05/2021 |          1608.72  |    408 |       0 |       63 |
| 30/06/2021 |          1108.46  |    370 |       0 |       63 |
| 31/07/2021 |          1059.52  |    398 |       0 |       84 |
| 31/08/2021 |           469.172 |    334 |       0 |       95 |
| 30/09/2021 |          1649.79  |    360 |       0 |       55 |
| 31/10/2021 |           457.177 |    341 |       0 |      104 |
| 30/11/2021 |          1869.66  |    426 |       0 |       77 |
| 31/12/2021 |          1107.44  |    391 |       0 |       72 |
| 31/01/2022 |          1092.29  |    366 |       0 |       71 |
| 28/02/2022 |           971.043 |    330 |       0 |       78 |
| 31/03/2022 |          1036.55  |    385 |       0 |       90 |
| 30/04/2022 |          1150.98  |    369 |       0 |       99 |
| 31/05/2022 |          1628.67  |    399 |       0 |       69 |
| 30/06/2022 |          1711.04  |    396 |       0 |       75 |
| 31/07/2022 |          1125.66  |    340 |       0 |       97 |
| 31/08/2022 |           623.937 |    333 |       0 |       89 |
| 30/09/2022 |           714.926 |    321 |       0 |       85 |
| 31/10/2022 |           550.908 |    323 |       0 |      100 |
| 30/11/2022 |           756.554 |    335 |       0 |       90 |
| 31/12/2022 |           507.909 |    309 |       0 |      112 |
| 31/01/2023 |           898.696 |    359 |       0 |      102 |
| 28/02/2023 |           470.232 |    305 |       0 |       80 |
| 31/03/2023 |            86.753 |    324 |       0 |      105 |
| 30/04/2023 |           396.627 |    335 |       0 |      100 |
| 31/05/2023 |           413.633 |    317 |       0 |       99 |
| 30/06/2023 |           614.541 |    325 |       0 |       97 |
| 31/07/2023 |           170.867 |    274 |       0 |      105 |
| 31/08/2023 |           140.953 |    138 |       0 |       56 |
=================== SUMMARY METRICS ===================
| Metric                      | Value                 |
|-----------------------------+-----------------------|
| Backtesting from            | 2021-01-02 06:00:00   |
| Backtesting to              | 2023-08-15 00:00:00   |
| Max open trades             | 4                     |
|                             |                       |
| Total/Daily Avg Trades      | 13787 / 14.45         |
| Starting balance            | 1000 USDT             |
| Final balance               | 32249.212 USDT        |
| Absolute profit             | 31249.212 USDT        |
| Total profit %              | 3124.92%              |
| CAGR %                      | 277.71%               |
| Sortino                     | 41.66                 |
| Sharpe                      | 49.69                 |
| Calmar                      | 5520.15               |
| Profit factor               | 1.71                  |
| Expectancy (Ratio)          | 2.27 (0.14)           |
| Trades per day              | 14.45                 |
| Avg. daily profit %         | 3.28%                 |
| Avg. stake amount           | 199.469 USDT          |
| Total trade volume          | 2750082.274 USDT      |
|                             |                       |
| Long / Short                | 7667 / 6120           |
| Total profit Long %         | 1870.19%              |
| Total profit Short %        | 1254.73%              |
| Absolute profit Long        | 18701.924 USDT        |
| Absolute profit Short       | 12547.288 USDT        |
|                             |                       |
| Best Pair                   | CRV/USDT:USDT 740.20% |
| Worst Pair                  | CHZ/USDT:USDT -93.77% |
| Best trade                  | BEL/USDT:USDT 12.60%  |
| Worst trade                 | ICX/USDT:USDT -36.50% |
| Best day                    | 408.354 USDT          |
| Worst day                   | -156.319 USDT         |
| Days win/draw/lose          | 641 / 0 / 308         |
| Avg. Duration Winners       | 0:59:00               |
| Avg. Duration Loser         | 5:03:00               |
| Max Consecutive Wins / Loss | 52 / 9                |
| Rejected Entry signals      | 24476                 |
| Entry/Exit Timeouts         | 0 / 2588              |
|                             |                       |
| Min balance                 | 1020.199 USDT         |
| Max balance                 | 32260.871 USDT        |
| Max % of account underwater | 12.62%                |
| Absolute Drawdown (Account) | 1.13%                 |
| Absolute Drawdown           | 296.897 USDT          |
| Drawdown high               | 25189.059 USDT        |
| Drawdown low                | 24892.162 USDT        |
| Drawdown Start              | 2022-08-09 05:00:00   |
| Drawdown End                | 2022-08-12 08:00:00   |
| Market change               | 26.08%                |
=======================================================

Backtested 2021-01-02 06:00:00 -> 2023-08-15 00:00:00 | Max open trades : 4
==================================================================================== STRATEGY SUMMARY ====================================================================================
|                           Strategy |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |            Drawdown |
|------------------------------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------+---------------------|
| CusTrend_coralTrend_Adx_EMA_Oct_1h |     13787 |           1.14 |       15659.12 |         31249.212 |        3124.92 |        1:46:00 | 11114     0  2673  80.6 | 296.897 USDT  1.13% |
==========================================================================================================================================================================================
    '''

    minimal_roi = {'0': 0.101, '373': 0.068, '1088': 0.025, '1336': 0}


    stoploss = -0.347

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30

    leverage_optimize = True
    leverage_num = IntParameter(low=1, high=5, default=4, space='buy', optimize=leverage_optimize)

    parameters_yes = True
    parameters_no = False

    '''
    
    '''
    adx_long_min = IntParameter(4, 21, default=21, space="buy", optimize = parameters_yes)
    adx_long_max = IntParameter(21, 56, default=50, space="buy", optimize = parameters_yes)
    adx_short_min = IntParameter(4, 21, default=13, space="buy", optimize = parameters_yes)
    adx_short_max = IntParameter(20, 56, default=51, space="buy", optimize = parameters_yes)

    ema_period = IntParameter(22, 200, default=142, space="buy", optimize= parameters_yes)

    volume_check = IntParameter(15, 45, default=22, space="buy", optimize= parameters_yes)
    volume_check_exit = IntParameter(15, 45, default=19, space="sell", optimize= parameters_yes)

    sell_shift = IntParameter(1, 6, default=5, space="sell", optimize= parameters_yes)
    sell_shift_short = IntParameter(1, 6, default=5, space="sell", optimize= parameters_yes)  # **added this at random-state 11165



    protect_optimize = True

    max_drawdown_lookback = IntParameter(1, 50, default=2, space="protection", optimize=protect_optimize)
    max_drawdown_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    max_drawdown_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)
    max_allowed_drawdown = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space="protection",
                                            optimize=protect_optimize)
    stoploss_guard_lookback = IntParameter(1, 50, default=8, space="protection", optimize=protect_optimize)
    stoploss_guard_trade_limit = IntParameter(1, 3, default=1, space="protection", optimize=protect_optimize)
    stoploss_guard_stop_duration = IntParameter(1, 50, default=4, space="protection", optimize=protect_optimize)

    @property
    def protections(self):
        return [




            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stoploss_guard_stop_duration.value,
                "only_per_pair": False
            }
        ]




    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    @property
    def plot_config(self):
        return {

            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {

                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '4h') for pair in pairs]
        
        return informative_pairs
    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:

            return dataframe
        
        inf_tf = '4h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        informative['ema'] = ta.EMA(informative['close'], timeperiod=50)





        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        L_determine_trend_strategy = determine_trend(df = dataframe)
        dataframe['trend'] = L_determine_trend_strategy['trend']

        dataframe['psar'] = ta.SAR(dataframe)

        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)

        dataframe['ema'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period.value)

        dataframe['volume_mean'] = dataframe['volume'].rolling(self.volume_check.value).mean().shift(1)
        dataframe['volume_mean_exit'] = dataframe['volume'].rolling(self.volume_check_exit.value).mean().shift(1)


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                        (dataframe['adx'] > self.adx_long_min.value) & # trend strength confirmation
                        (dataframe['adx'] < self.adx_long_max.value) & # trend strength confirmation

                        (dataframe['psar'] < dataframe['close']) & 
                        (dataframe['ema'] < dataframe['close']) & 
                        (dataframe['ema_4h'] < dataframe['close']) & 



                        (dataframe['trend'] == 1) &

                        (dataframe['rsi'] > 50) &

                        (dataframe['volume'] > dataframe['volume_mean'])

            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                        (dataframe['adx'] > self.adx_short_min.value) & # trend strength confirmation
                        (dataframe['adx'] < self.adx_short_max.value) & # trend strength confirmation

                        (dataframe['psar'] > dataframe['close']) & # trend reversal confirmation
                        (dataframe['ema'] > dataframe['close']) & # trend confirmation
                        (dataframe['ema_4h'] > dataframe['close']) & 




                        (dataframe['trend'] == -1) & 

                        (dataframe['rsi'] < 50) & # momentum indicator

                        (dataframe['volume'] > dataframe['volume_mean']) # volume weighted indicator
            ),
            'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''

        exit_long = (
                (dataframe['close'] < dataframe['low'].shift(self.sell_shift.value)) &
                (dataframe['volume'] > dataframe['volume_mean_exit'])
        )

        exit_short = (
                (dataframe['close'] > dataframe['high'].shift(self.sell_shift_short.value)) &
                (dataframe['volume'] > dataframe['volume_mean_exit'])
        )


        conditions_short.append(exit_short)
        dataframe.loc[exit_short, 'exit_tag'] += 'exit_short'


        conditions_long.append(exit_long)
        dataframe.loc[exit_long, 'exit_tag'] += 'exit_long'


        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'exit_long'] = 1

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'exit_short'] = 1
            
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value
    