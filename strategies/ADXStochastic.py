



import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)


import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


"""
Recursive analysis in freqtrade aims to validate trading strategies for inaccuracies due to recursive issues with certain indicators, ensuring reliable performance in both backtesting and live trading environments.
For more information, visit: https://www.freqtrade.io/en/stable/recursive-analysis/#recursive-analysis-command-reference

Command:
    freqtrade recursive-analysis -i 5m -s ADXStochastic -c user_data/config_bybit_futures_20.json --startup-candle 50 100 150 250 300 --timerange 20231010-20240410

    | indicators   | 50     | 100    | 150    | 250     | 300    |
    |--------------+--------+--------+--------+---------+--------|
    | adx          | 9.646% | 1.133% | 0.004% | -0.000% | 0.000% |
    | slowd        | 0.000% | 0.000% | 0.000% | 0.000%  | 0.000% |
    | slowk        | 0.000% | 0.000% | 0.000% | 0.000%  | 0.000% |

"""

"""
Command:
    freqtrade hyperopt -i 5m -s ADXStochastic -c user_data/config_bybit_futures_20.json --hyperopt-loss SharpeHyperOptLossDaily -e 86 --spaces roi stoploss buy sell --timerange 20231010-20240210 --job-workers 9 --random-state 9319 --strategy-path user_data/strategies

"""

"""
Command:
    freqtrade backtesting --strategy ADXStochastic --timeframe 5m --timeframe-detail 1m --timerange 20231010-20240410 --max-open-trades 5 --config user_data/config_bybit_futures_20.json --breakdown month 

    Result for strategy ADXStochastic
=============================================================== BACKTESTING REPORT ==============================================================
|            Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|-----------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
|   BCH/USDT:USDT |        86 |           2.22 |         190.63 |          6050.577 |         605.06 |        1:13:00 |    62     4    20  72.1 |
|   ETH/USDT:USDT |       107 |           1.03 |         109.82 |          5765.765 |         576.58 |        1:32:00 |    66     8    33  61.7 |
|   ETC/USDT:USDT |        74 |           2.59 |         191.89 |          5586.484 |         558.65 |        1:07:00 |    53     5    16  71.6 |
|   ADA/USDT:USDT |        74 |           3.74 |         276.72 |          5329.806 |         532.98 |        0:56:00 |    56     2    16  75.7 |
|   UNI/USDT:USDT |        74 |           1.90 |         140.59 |          4781.660 |         478.17 |        1:09:00 |    49     4    21  66.2 |
|  DOGE/USDT:USDT |        87 |           2.19 |         190.51 |          4136.348 |         413.63 |        1:00:00 |    63     2    22  72.4 |
|   LTC/USDT:USDT |        69 |           2.08 |         143.85 |          4060.491 |         406.05 |        1:13:00 |    49     4    16  71.0 |
|   MNT/USDT:USDT |       154 |          -0.08 |         -11.56 |          3708.978 |         370.90 |        1:33:00 |    84    11    59  54.5 |
|  LINK/USDT:USDT |        75 |           3.04 |         228.06 |          3682.158 |         368.22 |        1:02:00 |    54     2    19  72.0 |
|   BTC/USDT:USDT |       124 |           1.39 |         172.43 |          3461.233 |         346.12 |        1:26:00 |    80     6    38  64.5 |
|   TON/USDT:USDT |        69 |           1.44 |          99.68 |          2710.243 |         271.02 |        1:30:00 |    43     8    18  62.3 |
|   DOT/USDT:USDT |        72 |           3.07 |         220.86 |          1943.489 |         194.35 |        1:07:00 |    52     3    17  72.2 |
| MATIC/USDT:USDT |        73 |           0.39 |          28.46 |          1776.142 |         177.61 |        1:09:00 |    47     5    21  64.4 |
|   BNB/USDT:USDT |        89 |           1.55 |         137.99 |          1111.481 |         111.15 |        1:17:00 |    61     3    25  68.5 |
|  AVAX/USDT:USDT |        99 |           1.45 |         143.78 |           993.869 |          99.39 |        0:58:00 |    68     2    29  68.7 |
|   XRP/USDT:USDT |       103 |           0.81 |          83.46 |           223.725 |          22.37 |        1:20:00 |    64     4    35  62.1 |
|  ATOM/USDT:USDT |        69 |          -0.01 |          -0.48 |          -485.642 |         -48.56 |        1:15:00 |    40     5    24  58.0 |
|   SOL/USDT:USDT |        74 |           0.71 |          52.52 |          -524.553 |         -52.46 |        0:54:00 |    48     2    24  64.9 |
|  NEAR/USDT:USDT |        68 |          -0.36 |         -24.31 |         -1915.271 |        -191.53 |        0:47:00 |    42     0    26  61.8 |
|   TRX/USDT:USDT |       127 |          -0.54 |         -69.16 |         -2798.504 |        -279.85 |        2:09:00 |    71     9    47  55.9 |
|           TOTAL |      1767 |           1.30 |        2305.72 |         49598.481 |        4959.85 |        1:17:00 |  1152    89   526  65.2 |
======================================================= LEFT OPEN TRADES REPORT ========================================================
|   Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|--------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
|  TOTAL |         0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
=========================================================== ENTER TAG STATS ===========================================================
|   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|-------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
| TOTAL |      1767 |           1.30 |        2305.72 |         49598.481 |        4959.85 |        1:17:00 |  1152    89   526  65.2 |
===================================================== EXIT REASON STATS =====================================================
|   Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|---------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
|           roi |    1195 |   1080    89    26  90.4 |           6.25 |        7467.48 |          177106   |        1493.5  |
|   exit_signal |     323 |     72     0   251  22.3 |          -2.46 |        -794.3  |          -13943.6 |        -158.86 |
|     stop_loss |     249 |      0     0   249     0 |         -17.54 |       -4367.46 |         -113564   |        -873.49 |
======================= MONTH BREAKDOWN ========================
|      Month |   Tot Profit USDT |   Wins |   Draws |   Losses |
|------------+-------------------+--------+---------+----------|
| 31/10/2023 |           888.297 |    155 |      15 |       66 |
| 30/11/2023 |          2354.5   |    205 |      16 |       89 |
| 31/12/2023 |          3229.04  |    201 |      19 |      106 |
| 31/01/2024 |          -170.425 |    193 |      14 |       89 |
| 29/02/2024 |          3539.43  |    132 |      16 |       74 |
| 31/03/2024 |         28195.4   |    206 |       5 |       81 |
| 30/04/2024 |         11562.2   |     60 |       4 |       21 |
=================== SUMMARY METRICS ====================
| Metric                      | Value                  |
|-----------------------------+------------------------|
| Backtesting from            | 2023-10-10 00:00:00    |
| Backtesting to              | 2024-04-10 00:00:00    |
| Max open trades             | 5                      |
|                             |                        |
| Total/Daily Avg Trades      | 1767 / 9.66            |
| Starting balance            | 1000 USDT              |
| Final balance               | 50598.481 USDT         |
| Absolute profit             | 49598.481 USDT         |
| Total profit %              | 4959.85%               |
| CAGR %                      | 250489.42%             |
| Sortino                     | 13.75                  |
| Sharpe                      | 15.17                  |
| Calmar                      | 1980.19                |
| Profit factor               | 1.38                   |
| Expectancy (Ratio)          | 28.07 (0.06)           |
| Trades per day              | 9.66                   |
| Avg. daily profit %         | 27.10%                 |
| Avg. stake amount           | 2111.598 USDT          |
| Total trade volume          | 3731194.397 USDT       |
|                             |                        |
| Best Pair                   | ADA/USDT:USDT 276.72%  |
| Worst Pair                  | TRX/USDT:USDT -69.16%  |
| Best trade                  | LINK/USDT:USDT 23.60%  |
| Worst trade                 | LINK/USDT:USDT -18.16% |
| Best day                    | 9974.882 USDT          |
| Worst day                   | -5523.065 USDT         |
| Days win/draw/lose          | 128 / 1 / 54           |
| Avg. Duration Winners       | 0:56:00                |
| Avg. Duration Loser         | 1:51:00                |
| Max Consecutive Wins / Loss | 29 / 14                |
| Rejected Entry signals      | 418                    |
| Entry/Exit Timeouts         | 0 / 0                  |
|                             |                        |
| Min balance                 | 986.084 USDT           |
| Max balance                 | 51243.46 USDT          |
| Max % of account underwater | 37.38%                 |
| Absolute Drawdown (Account) | 26.15%                 |
| Absolute Drawdown           | 11985.157 USDT         |
| Drawdown high               | 44833.832 USDT         |
| Drawdown low                | 32848.675 USDT         |
| Drawdown Start              | 2024-03-22 12:35:00    |
| Drawdown End                | 2024-03-24 00:07:00    |
| Market change               | 197.15%                |
========================================================

Backtested 2023-10-10 00:00:00 -> 2024-04-10 00:00:00 | Max open trades : 5
=========================================================================== STRATEGY SUMMARY ===========================================================================
|      Strategy |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |               Drawdown |
|---------------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------+------------------------|
| ADXStochastic |      1767 |           1.30 |        2305.72 |         49598.481 |        4959.85 |        1:17:00 |  1152    89   526  65.2 | 11985.157 USDT  26.15% |
========================================================================================================================================================================
"""

"""
Command:
    freqtrade plot-profit -i 5m --timerange 20231010-20240410 --export-filename user_data/backtest_results/backtest-result-2024-05-01_04-57-58.json -c user_data/config_bybit_futures_20.json 

"""

"""
Command:
    freqtrade webserver --config user_data/config_bybit_futures_20.json
"""

class ADXStochastic(IStrategy):


    INTERFACE_VERSION = 3

    timeframe = '5m'

    can_short: bool = False


    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }


    stoploss = -0.10

    trailing_stop = False




    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 150

    buy_adx_threshold = IntParameter(low=45, high=55, default=50, space='buy', optimize=True)
    buy_stoch_threshold = IntParameter(low=15, high=25, default=20, space='buy', optimize=True)

    sell_adx_threshold = IntParameter(low=20, high=30, default=25, space='sell', optimize=True)
    sell_stoch_threshold = IntParameter(low=70, high=80, default=75, space='sell', optimize=True)

    leverage_level = 9

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
            },
            'subplots': {

                "ADX": {
                    'adx': {'color': 'rgb(255, 82, 82)'}, 
                },
                "Stochastic": {
                    'fastk': {'color': 'rgb(41, 98, 255)'},
                    'fastd': {'color': 'rgb(255, 109, 0)'},
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['adx'] = ta.ADX(dataframe)





        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        dataframe.loc[
            (
                (dataframe['adx'] > self.buy_adx_threshold.value) &
                (dataframe['fastk'].shift(1) < self.buy_stoch_threshold.value) &
                (dataframe['fastd'].shift(1) < self.buy_stoch_threshold.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        dataframe.loc[
            (
                (dataframe['adx'] < self.sell_adx_threshold.value) & 
                (dataframe['fastk'] > self.sell_stoch_threshold.value) & 
                (dataframe['fastk'] > self.sell_stoch_threshold.value) & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level
    