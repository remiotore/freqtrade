# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, IntParameter, DecimalParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, List, Optional, Union, Tuple
import talib.abstract as ta
from technical import qtpylib
import numpy as np

class ZTV16X(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = False

    # ROI table:
    minimal_roi = {
        "0": 0.20,  # 20% ROI for any trade
        "30": 0.10, # Reduce ROI to 10% after 30 minutes
        "60": 0.05, # Reduce ROI to 5% after 60 minutes
        "120": 0    # Exit after 120 minutes
    }

    # Stoploss:
    stoploss = -0.15

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.107
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 7

    # Hyperparameters for optimization
    adx_threshold = IntParameter(20, 40, default=25, space='buy', optimize=True)
    mfi_threshold_long = IntParameter(50, 70, default=60, space='buy', optimize=True)
    mfi_threshold_short = IntParameter(30, 50, default=40, space='sell', optimize=True)
    bollinger_std_dev = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {}
        plot_config['subplots'] = {
            "DI": {
                'dx' : {'color': 'yellow'},
                'adx': {'color': 'orange'},
                'pdi': {'color': 'green'},
                'mdi': {'color': 'red'},
            },
            "AROON": {
                'aup': { 'color': 'green' },
                'ado': { 'color': 'red' }
            },
        }
        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ADX and DI indicators
        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)

        # MFI (Money Flow Index)
        dataframe['mfi'] = ta.MFI(dataframe)

        # AROON indicator
        dataframe[['aup', 'ado']] = ta.AROON(dataframe)[['aroonup','aroondown']]

        # Bollinger Bands with dynamic standard deviation
        dataframe[['bbl','bbm','bbu']] = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=8, stds=self.bollinger_std_dev.value
        )[['lower','mid','upper']]

        # ATR (Average True Range) for volatility-based position sizing
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # RSI (Relative Strength Index) for overbought/oversold confirmation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # EMA (Exponential Moving Average) for trend confirmation
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry conditions
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['bbu'])) &  # Price crosses above upper Bollinger Band
                (dataframe['adx'] > self.adx_threshold.value) &  # ADX above threshold (strong trend)
                (dataframe['mfi'] > self.mfi_threshold_long.value) &  # MFI above threshold (strong buying pressure)
                (dataframe['rsi'] < 70) &  # RSI not overbought
                (dataframe['ema_50'] > dataframe['ema_200']) &  # Short-term EMA above long-term EMA (uptrend)
                (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean())  # Volume above average
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long Bollinger enter')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &  # DX > MDI
                (dataframe['adx'] > self.adx_threshold.value) &  # ADX above threshold
                (dataframe['pdi'] > dataframe['mdi']) &  # PDI > MDI
                (dataframe['aup'] > dataframe['ado']) &  # AROON up > AROON down
                (dataframe['aup'] > 50) &  # AROON up > 50
                (dataframe['mfi'] > self.mfi_threshold_long.value) &  # MFI above threshold
                (dataframe['rsi'] < 70) &  # RSI not overbought
                (dataframe['ema_50'] > dataframe['ema_200']) &  # Short-term EMA above long-term EMA (uptrend)
                (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean())  # Volume above average
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')

        # Short entry conditions
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['bbl'])) &  # Price crosses below lower Bollinger Band
                (dataframe['adx'] > self.adx_threshold.value) &  # ADX above threshold (strong trend)
                (dataframe['mfi'] < self.mfi_threshold_short.value) &  # MFI below threshold (strong selling pressure)
                (dataframe['rsi'] > 30) &  # RSI not oversold
                (dataframe['ema_50'] < dataframe['ema_200']) &  # Short-term EMA below long-term EMA (downtrend)
                (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean())  # Volume above average
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short Bollinger enter')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['pdi']) &  # DX > PDI
                (dataframe['adx'] > self.adx_threshold.value) &  # ADX above threshold
                (dataframe['mdi'] > dataframe['pdi']) &  # MDI > PDI
                (dataframe['ado'] > dataframe['aup']) &  # AROON down > AROON up
                (dataframe['ado'] > 50) &  # AROON down > 50
                (dataframe['mfi'] < self.mfi_threshold_short.value) &  # MFI below threshold
                (dataframe['rsi'] > 30) &  # RSI not oversold
                (dataframe['ema_50'] < dataframe['ema_200']) &  # Short-term EMA below long-term EMA (downtrend)
                (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean())  # Volume above average
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long positions
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['bbm'])) &  # Price crosses below middle Bollinger Band
                (dataframe['mfi'] < self.mfi_threshold_long.value) &  # MFI below threshold (weak buying pressure)
                (dataframe['rsi'] < 50)  # RSI below 50 (momentum weakening)
            ),
            'exit_long'
        ] = 1

        # Exit short positions
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['bbm'])) &  # Price crosses above middle Bollinger Band
                (dataframe['mfi'] > self.mfi_threshold_short.value) &  # MFI above threshold (weak selling pressure)
                (dataframe['rsi'] > 50)  # RSI above 50 (momentum weakening)
            ),
            'exit_short'
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 5  # Reduced leverage to 5x

    def custom_stake_amount(self, pair: str, current_time, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        # Dynamic position sizing based on ATR (volatility)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        atr = last_candle['atr']

        # Adjust stake size based on volatility and account equity
        stake_size = max(min_stake, min(max_stake, proposed_stake * (1.0 / atr)))
        return stake_size
