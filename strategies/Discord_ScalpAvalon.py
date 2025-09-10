# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import ta.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class ScalpAvalon(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.01,
        "336": 0.005,
        "672": 0.0025,
        "1344": 0
    }

    stoploss = -0.10
    timeframe = '5m'
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_custom_stoploss = True
    buy_params = {
        "base_nb_candles_buy": 13,
        "ewo_high": 5.0,
        "ewo_low": -8.0,
        "low_offset": 0.975,
        "rsi_buy": 34.0,
        "ma_period": 20,
        "ma_slope": 0.0015,
        "volume_slope": 0.001
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 18,
        "low_offset": 0.975,
        "high_offset": 1.015,
        "rsi_sell": 67.0
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add EMA indicators
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # Add RSI indicator
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Add MACD indicator
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Add Fibonacci retracement levels
        high = dataframe['high'].max()
        low = dataframe['low'].min()
        dataframe['fib38'] = high - 0.382 * (high - low)
        dataframe['fib50'] = high - 0.5 * (high - low)
        dataframe['fib62'] = high - 0.618 * (high - low)

        # Add Bollinger Bands
        bollingerbands = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_upper'] = bollingerbands['upperband']
        dataframe['bb_middle'] = bollingerbands['middleband']
        dataframe['bb_lower'] = bollingerbands['lowerband']

        # Add Stochastic Oscillator
        stoch = ta.STOCH(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # Add ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Add Volume Slope
        dataframe['vol_slope'] = qtpylib.slope(dataframe['volume'], period=self.timeframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI oversold condition
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_oversold'] = (dataframe['rsi'] < 30).astype(int)

        # MACD bullish crossover
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macd_crossover'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])

        # Fibonacci retracement support
        dataframe['fib_0_382'] = ta.FIBONACCI(dataframe, retracement_level=0.382)
        dataframe['fib_0_5'] = ta.FIBONACCI(dataframe, retracement_level=0.5)
        dataframe['fib_0_618'] = ta.FIBONACCI(dataframe, retracement_level=0.618)
        dataframe['fib_0_786'] = ta.FIBONACCI(dataframe, retracement_level=0.786)
        dataframe['fib_support'] = ((dataframe['low'] <= dataframe['fib_0_382']) & (dataframe['close'] > dataframe['fib_0_382'])) | \
                                   ((dataframe['low'] <= dataframe['fib_0_5']) & (dataframe['close'] > dataframe['fib_0_5'])) | \
                                   ((dataframe['low'] <= dataframe['fib_0_618']) & (dataframe['close'] > dataframe['fib_0_618'])) | \
                                   ((dataframe['low'] <= dataframe['fib_0_786']) & (dataframe['close'] > dataframe['fib_0_786']))

        # Simple Moving Average slope
        ma_slope = qtpylib.slope(dataframe['close'], period=self.buy_params['ma_period'])
        dataframe['ma_signal'] = (ma_slope > self.buy_params['ma_slope']).astype(int)

        # Volume slope
        dataframe['vol_signal'] = (dataframe['vol_slope'] > self.buy_params['volume_slope']).astype(int)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['fib_resistance'] = ((dataframe['high'] >= dataframe['fib_0_5']) & (dataframe['close'] < dataframe['fib_0_5'])) | \
                                      ((dataframe['high'] >= dataframe['fib_0_618']) & (dataframe['close'] < dataframe['fib_0_618'])) | \
                                      ((dataframe['high'] >= dataframe['fib_0_786']) & (dataframe['close'] < dataframe['fib_0_786']))

        return dataframe
