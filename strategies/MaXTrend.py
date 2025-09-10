# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from functools import reduce


from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


"""

freqtrade backtesting --strategy MaXTrend --timeframe 4h --timeframe-detail 30m --timerange 20230901-20240901 --breakdown month -c user_data/config_bybit_futures.json

Result for strategy MaXTrend
                                                  BACKTESTING REPORT                                                   
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃     Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│  SOL/USDT:USDT │     20 │         6.74 │         413.735 │        41.37 │ 4 days, 20:24:00 │   12     0     8  60.0 │
│  FTM/USDT:USDT │     13 │         5.18 │         386.127 │        38.61 │ 4 days, 20:18:00 │    8     0     5  61.5 │
│ NEAR/USDT:USDT │     18 │         5.03 │         357.189 │        35.72 │ 5 days, 15:07:00 │    7     0    11  38.9 │
│ AVAX/USDT:USDT │     14 │        10.68 │         330.021 │         33.0 │ 5 days, 14:51:00 │    7     0     7  50.0 │
│  GRT/USDT:USDT │     13 │         5.05 │         287.685 │        28.77 │ 5 days, 13:14:00 │    9     0     4  69.2 │
│  FIL/USDT:USDT │     16 │          3.2 │         271.546 │        27.15 │ 4 days, 13:00:00 │    6     0    10  37.5 │
│  ADA/USDT:USDT │     15 │         4.59 │         243.274 │        24.33 │ 4 days, 23:12:00 │    9     0     6  60.0 │
│  ICP/USDT:USDT │     16 │         5.05 │         227.493 │        22.75 │ 3 days, 12:00:00 │    5     0    11  31.2 │
│  UNI/USDT:USDT │     12 │         1.93 │         138.390 │        13.84 │ 3 days, 20:20:00 │    5     0     7  41.7 │
│ ALGO/USDT:USDT │     12 │         0.88 │         129.816 │        12.98 │  4 days, 3:20:00 │    5     0     7  41.7 │
│  BNB/USDT:USDT │     23 │         0.92 │          85.952 │          8.6 │ 3 days, 16:21:00 │   11     0    12  47.8 │
│  BTC/USDT:USDT │     25 │         1.68 │          82.299 │         8.23 │ 4 days, 19:02:00 │    9     0    16  36.0 │
│ AAVE/USDT:USDT │     13 │         0.36 │          75.788 │         7.58 │  4 days, 7:05:00 │    5     0     8  38.5 │
│ ATOM/USDT:USDT │     14 │        -0.18 │          -2.854 │        -0.29 │  3 days, 2:00:00 │    5     0     9  35.7 │
│  XRP/USDT:USDT │     14 │          0.5 │         -11.324 │        -1.13 │ 3 days, 17:43:00 │    6     0     8  42.9 │
│  DOT/USDT:USDT │     17 │        -0.45 │         -17.127 │        -1.71 │ 3 days, 15:18:00 │    5     0    12  29.4 │
│  ETH/USDT:USDT │     13 │        -0.51 │         -51.163 │        -5.12 │  4 days, 1:51:00 │    3     0    10  23.1 │
│ LINK/USDT:USDT │     16 │        -0.52 │         -79.698 │        -7.97 │  3 days, 9:30:00 │    5     0    11  31.2 │
│  LTC/USDT:USDT │     11 │        -2.71 │        -144.288 │       -14.43 │  2 days, 8:00:00 │    1     0    10   9.1 │
│          TOTAL │    295 │         2.58 │        2722.862 │       272.29 │  4 days, 6:39:00 │  123     0   172  41.7 │
└────────────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────────┴────────────────────────┘
                                               LEFT OPEN TRADES REPORT                                                
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ATOM/USDT:USDT │      1 │         0.21 │           1.526 │         0.15 │         8:00:00 │    1     0     0   100 │
│  FIL/USDT:USDT │      1 │        -0.24 │          -1.781 │        -0.18 │         4:00:00 │    0     0     1     0 │
│  DOT/USDT:USDT │      1 │        -0.31 │          -2.271 │        -0.23 │ 2 days, 0:00:00 │    0     0     1     0 │
│  SOL/USDT:USDT │      1 │        -0.41 │          -2.951 │         -0.3 │         4:00:00 │    0     0     1     0 │
│  XRP/USDT:USDT │      1 │        -0.87 │          -6.418 │        -0.64 │ 2 days, 0:00:00 │    0     0     1     0 │
│          TOTAL │      5 │        -0.32 │         -11.895 │        -1.19 │        22:24:00 │    1     0     4  20.0 │
└────────────────┴────────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┘
                                                 ENTER TAG STATS                                                  
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Enter Tag ┃ Entries ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│     OTHER │     295 │         2.58 │        2722.862 │       272.29 │ 4 days, 6:39:00 │  123     0   172  41.7 │
│     TOTAL │     295 │         2.58 │        2722.862 │       272.29 │ 4 days, 6:39:00 │  123     0   172  41.7 │
└───────────┴─────────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┘
                                                EXIT REASON STATS                                                 
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Exit Reason ┃ Exits ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ exit_signal │   290 │         2.64 │        2734.757 │       273.48 │ 4 days, 8:02:00 │  122     0   168  42.1 │
│  force_exit │     5 │        -0.32 │         -11.895 │        -1.19 │        22:24:00 │    1     0     4  20.0 │
│       TOTAL │   295 │         2.58 │        2722.862 │       272.29 │ 4 days, 6:39:00 │  123     0   172  41.7 │
└─────────────┴───────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┘
                                                        MIXED TAG STATS                                                        
┏━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Enter Tag ┃ Exit Reason ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│           │ exit_signal │    290 │         2.64 │        2734.757 │       273.48 │ 4 days, 8:02:00 │  122     0   168  42.1 │
│           │  force_exit │      5 │        -0.32 │         -11.895 │        -1.19 │        22:24:00 │    1     0     4  20.0 │
│     TOTAL │             │    295 │         2.58 │        2722.862 │       272.29 │ 4 days, 6:39:00 │  123     0   172  41.7 │
└───────────┴─────────────┴────────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┘
                    MONTH BREAKDOWN                     
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃      Month ┃ Tot Profit USDT ┃ Wins ┃ Draws ┃ Losses ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ 30/09/2023 │         -39.968 │    8 │     0 │     21 │
│ 31/10/2023 │         113.042 │   13 │     0 │     12 │
│ 30/11/2023 │         280.122 │   10 │     0 │     10 │
│ 31/12/2023 │         666.172 │   13 │     0 │     15 │
│ 31/01/2024 │         132.629 │   11 │     0 │     11 │
│ 29/02/2024 │         258.883 │    8 │     0 │     12 │
│ 31/03/2024 │         505.907 │   12 │     0 │      9 │
│ 30/04/2024 │         400.819 │   11 │     0 │     10 │
│ 31/05/2024 │        -243.874 │   10 │     0 │     23 │
│ 30/06/2024 │         345.846 │   10 │     0 │     13 │
│ 31/07/2024 │        -201.542 │    4 │     0 │     21 │
│ 31/08/2024 │          516.72 │   12 │     0 │     11 │
│ 30/09/2024 │         -11.895 │    1 │     0 │      4 │
└────────────┴─────────────────┴──────┴───────┴────────┘
                    SUMMARY METRICS                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                      ┃ Value                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Backtesting from            │ 2023-09-01 00:00:00    │
│ Backtesting to              │ 2024-09-01 00:00:00    │
│ Max open trades             │ 5                      │
│                             │                        │
│ Total/Daily Avg Trades      │ 295 / 0.81             │
│ Starting balance            │ 1000 USDT              │
│ Final balance               │ 3722.862 USDT          │
│ Absolute profit             │ 2722.862 USDT          │
│ Total profit %              │ 272.29%                │
│ CAGR %                      │ 270.95%                │
│ Sortino                     │ 9.53                   │
│ Sharpe                      │ 3.22                   │
│ Calmar                      │ 128.48                 │
│ Profit factor               │ 2.00                   │
│ Expectancy (Ratio)          │ 9.23 (0.58)            │
│ Avg. daily profit %         │ 0.74%                  │
│ Avg. stake amount           │ 454.559 USDT           │
│ Total trade volume          │ 134094.768 USDT        │
│                             │                        │
│ Long / Short                │ 160 / 135              │
│ Total profit Long %         │ 169.27%                │
│ Total profit Short %        │ 103.01%                │
│ Absolute profit Long        │ 1692.747 USDT          │
│ Absolute profit Short       │ 1030.115 USDT          │
│                             │                        │
│ Best Pair                   │ AVAX/USDT:USDT 33.00%  │
│ Worst Pair                  │ LTC/USDT:USDT -14.43%  │
│ Best trade                  │ AVAX/USDT:USDT 88.73%  │
│ Worst trade                 │ AVAX/USDT:USDT -16.28% │
│ Best day                    │ 444.424 USDT           │
│ Worst day                   │ -111.863 USDT          │
│ Days win/draw/lose          │ 75 / 189 / 99          │
│ Avg. Duration Winners       │ 7 days, 0:20:00        │
│ Avg. Duration Loser         │ 2 days, 7:42:00        │
│ Max Consecutive Wins / Loss │ 8 / 19                 │
│ Rejected Entry signals      │ 665                    │
│ Entry/Exit Timeouts         │ 0 / 0                  │
│                             │                        │
│ Min balance                 │ 949.308 USDT           │
│ Max balance                 │ 3764.208 USDT          │
│ Max % of account underwater │ 11.06%                 │
│ Absolute Drawdown (Account) │ 11.06%                 │
│ Absolute Drawdown           │ 373.248 USDT           │
│ Drawdown high               │ 2373.865 USDT          │
│ Drawdown low                │ 2000.616 USDT          │
│ Drawdown Start              │ 2024-05-03 20:00:00    │
│ Drawdown End                │ 2024-06-12 20:00:00    │
│ Market change               │ 57.60%                 │
└─────────────────────────────┴────────────────────────┘

Backtested 2023-09-01 00:00:00 -> 2024-09-01 00:00:00 | Max open trades : 5
                                                           STRATEGY SUMMARY                                                            
┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Strategy ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃             Drawdown ┃
┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ MaXTrend │    295 │         2.58 │        2722.862 │       272.29 │ 4 days, 6:39:00 │  123     0   172  41.7 │ 373.248 USDT  11.06% │
└──────────┴────────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┴──────────────────────┘

"""


class MaXTrend(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.30

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Strategy parameters
    short_ma = IntParameter(5, 15, default=5, space='buy')
    mid_ma = IntParameter(20, 50, default=25, space='buy')
    long_ma = IntParameter(70, 150, default=75, space='buy')

    @property
    def plot_config(self):
        return {
            "main_plot": {
                f'short_ma_{self.short_ma.value}': {"color": "rgba(41, 98, 255, 0.65)"},
                f'mid_ma_{self.mid_ma.value}': {"color": "rgba(255, 235, 59, 0.65)"},
                f'long_ma_{self.long_ma.value}': {"color": "rgba(76, 175, 80, 0.65)"},
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

        for val in self.short_ma.range:
            dataframe[f'short_ma_{val}'] = ta.SMA(dataframe, timeperiod=val)
        
        for val in self.mid_ma.range:
            dataframe[f'mid_ma_{val}'] = ta.SMA(dataframe, timeperiod=val)
        
        for val in self.long_ma.range:
            dataframe[f'long_ma_{val}'] = ta.SMA(dataframe, timeperiod=val)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['low'], dataframe[f'short_ma_{self.short_ma.value}'])) &
                (dataframe[f'short_ma_{self.short_ma.value}'] > dataframe[f'mid_ma_{self.mid_ma.value}']) &
                (dataframe[f'mid_ma_{self.mid_ma.value}'] > dataframe[f'long_ma_{self.long_ma.value}']) &
                (dataframe[f'short_ma_{self.short_ma.value}'] > dataframe[f'short_ma_{self.short_ma.value}'].shift(1)) &
                (dataframe[f'mid_ma_{self.mid_ma.value}'] > dataframe[f'mid_ma_{self.mid_ma.value}'].shift(1)) &
                (dataframe[f'long_ma_{self.long_ma.value}'] > dataframe[f'long_ma_{self.long_ma.value}'].shift(1)) &
                (dataframe['volume'] > 0) 
            ),
            'enter_long'] = 1
        

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['high'], dataframe[f'short_ma_{self.short_ma.value}'])) &
                (dataframe[f'short_ma_{self.short_ma.value}'] < dataframe[f'mid_ma_{self.mid_ma.value}']) &
                (dataframe[f'mid_ma_{self.mid_ma.value}'] < dataframe[f'long_ma_{self.long_ma.value}']) &
                (dataframe[f'short_ma_{self.short_ma.value}'] < dataframe[f'short_ma_{self.short_ma.value}'].shift(1)) &
                (dataframe[f'mid_ma_{self.mid_ma.value}'] < dataframe[f'mid_ma_{self.mid_ma.value}'].shift(1)) &
                (dataframe[f'long_ma_{self.long_ma.value}'] < dataframe[f'long_ma_{self.long_ma.value}'].shift(1)) &
                (dataframe['volume'] > 0) 
            ),
            'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        

        dataframe.loc[
            (
                (dataframe['high'] < dataframe[f'mid_ma_{self.mid_ma.value}']) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['low'] > dataframe[f'mid_ma_{self.mid_ma.value}']) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'] = 1

        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return 1