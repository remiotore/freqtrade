# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.strategy import IStrategy
from datetime import datetime
from pandas import DataFrame
from typing import Dict, List
import talib.abstract as ta
from technical import qtpylib
from sklearn.preprocessing import MinMaxScaler



class ZaratustraV18(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = False
    exit_profit_only = True
    
    # ROI table:
    
    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }

    # Stoploss:
    stoploss = -0.20

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.050
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 10

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 6
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {
            'tsf' : { 'color' : 'black' },
        }
        plot_config['subplots'] = {
            'DI': {
                'dx' : { 'color': 'yellow' },
                'adx': { 'color': 'orange' },
                'pdi': { 'color': 'green' },
                'mdi': { 'color': 'red' },
                'atr': { 'color': 'purple' },
            },
            'Regresion' : {
                'slope' : {},
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['dx']    = ta.DX(dataframe)
        dataframe['adx']   = ta.ADX(dataframe)
        dataframe['pdi']   = ta.PLUS_DI(dataframe)
        dataframe['mdi']   = ta.MINUS_DI(dataframe)
        dataframe['tsf']   = ta.TSF(dataframe)
        dataframe['atr']   = MinMaxScaler(feature_range=(0, 100)).fit_transform(ta.ATR(dataframe).values.reshape(-1, 1))
        dataframe['slope'] = ta.LINEARREG_SLOPE(dataframe)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ##############################
        # Bollinger Bands Conditions #
        ##############################

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband']) &
                (dataframe['slope'] > dataframe['slope'].shift(1)) &
                (dataframe['slope'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long Bollinger enter')

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband']) &
                (dataframe['slope'] < dataframe['slope'].shift(1)) &
                (dataframe['slope'] < 0)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short Bollinger enter')

        ##################################
        # TimeSeries Forecast Conditions #
        ##################################

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['tsf']) &
                (dataframe['slope'] > dataframe['slope'].shift(1)) &
                (dataframe['slope'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long TimeSeries Forecast enter')

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['close'], dataframe['tsf']) &
                (dataframe['slope'] < dataframe['slope'].shift(1)) &
                (dataframe['slope'] < 0)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short TimeSeries Forecast enter')

        ####################################
        # Directional Indicator Conditions #
        ####################################

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['mdi']) &
                (dataframe['pdi'] > dataframe['mdi']) &
                (dataframe['atr'] > dataframe['mdi']) &
                (dataframe['slope'] > dataframe['slope'].shift(1)) &
                (dataframe['slope'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['pdi']) &
                (dataframe['adx'] > dataframe['pdi']) &
                (dataframe['mdi'] > dataframe['pdi']) &
                (dataframe['atr'] > dataframe['pdi']) &
                (dataframe['slope'] < dataframe['slope'].shift(1)) &
                (dataframe['slope'] < 0)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10