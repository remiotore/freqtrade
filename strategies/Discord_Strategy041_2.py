# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.persistence import Trade
from typing import Dict, List
from functools import reduce
from datetime import datetime, timedelta
from pandas import DataFrame, Series, DatetimeIndex, merge
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa

def WaveTrend(dataframe, chlen=10, avg=21, smalen=4):
    """
    WaveTrend Ocillator by LazyBear
    https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
    """
    df = dataframe.copy()

    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['esa'] = ta.EMA(df['hlc3'], timeperiod=chlen)
    df['d'] = ta.EMA((df['hlc3'] - df['esa']).abs(), timeperiod=chlen)
    df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'])
    df['tci'] = ta.EMA(df['ci'], timeperiod=avg)

    df['wt1'] = df['tci']
    df['wt2'] = ta.SMA(df['wt1'], timeperiod=smalen)
    df['wt1-wt2'] = df['wt1'] - df['wt2']

    return df['wt1'], df['wt2']

class Strategy041_2(IStrategy):
    """
    Strategy 041_2
    author@: Thy
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy005
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.05,
        "10": 0.04,
        "20": 0.03,
        "30": 0.02,
        "40": 0.01,
        "50": 0.005,
        "60": 0.0025,
        "100": 0.001,
        "120" : 0
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    #Custom Information
    custom_info = {}

    use_custom_stoploss = False

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=180) > trade.open_date:
            return -0.05
        elif current_time - timedelta(minutes=60) > trade.open_date:
            return -0.10
        return -0.10

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        pairs.append("BTC/USDT")
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        #dataframe = self.resample(dataframe, self.timeframe, 5)

        #print(DataFrame)

        # Wave Trend
        wt1, wt2 = WaveTrend(dataframe)

        dataframe['wave1'] = wt1
        dataframe['wave2'] = wt2

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        #dataframe['bb_upperband'] = bollinger['upper']

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Minus Directional Indicator / Movement
        #dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)

        #Mom Indicator
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)
        dataframe['mom_trend'] = dataframe['mom'].lt(dataframe['mom'].shift())

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        #AO
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # Overlap Studies
        # ------------------------------------

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma_40'] = ta.SMA(dataframe, timeperiod=40)
        dataframe['sma_100'] = ta.SMA(dataframe, timeperiod=100)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            # Prod
            (
                (dataframe['adx'] > 30) &
                (dataframe['fastd'] > dataframe['fastk']) &
		        (dataframe['close'] < dataframe['sma_40']) &
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['mfi'] < 34)
            ) |
            (
                (dataframe['adx'] > 16) &
                (dataframe['fastd'] > dataframe['fastk']) &
		        (dataframe['close'] < dataframe['sma_40']) &
                (dataframe['close'] > dataframe['sma_100']) &
                (dataframe['close'] < dataframe['bb_lowerband']) &
                (dataframe['mfi'] < 38)
            ) | (
                (qtpylib.crossed_above(dataframe['wave1'], dataframe['wave2'])) &
                (dataframe['wave1'] < -50) &
                (dataframe['wave2'] < -50)
            ), 
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            # Prod
            (
                (dataframe['macd'] < 0) &
                (dataframe['ao'] > 0) &
                (dataframe['ao'] < dataframe['ao'].shift())

            ) | (

                (dataframe['minus_di'] > 0) &
                (dataframe['ao'] > 0) &
                (dataframe['ao'] < dataframe['ao'].shift())
                
            ) | (
                (dataframe['sar'] > dataframe['close']) &
                (dataframe['fisher_rsi'] > 0.3)
            ),

            'sell'] = 1
        return dataframe

    def chaikin_mf(self, df, periods=20):
        close = df['close']
        low = df['low']
        high = df['high']
        volume = df['volume']

        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= volume
        cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()

        return Series(cmf, name='cmf')

    def resample(self, dataframe, interval, factor):
        # defines the reinforcement logic
        # resampled dataframe to establish if we are in an uptrend, downtrend or sideways trend
        df = dataframe.copy()
        df = df.set_index(DatetimeIndex(df['date']))
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        df = df.resample(str(int(interval[:-1]) * factor) + 'min', label="right").agg(ohlc_dict)
        df['resample_sma'] = ta.SMA(df, timeperiod=100, price='close')
        df['resample_medium'] = ta.SMA(df, timeperiod=50, price='close')
        df['resample_short'] = ta.SMA(df, timeperiod=25, price='close')
        df['resample_long'] = ta.SMA(df, timeperiod=200, price='close')
        df = df.drop(columns=['open', 'high', 'low', 'close'])
        df = df.resample(interval[:-1] + 'min')
        df = df.interpolate(method='time')
        df['date'] = df.index
        df.index = range(len(df))
        dataframe = merge(dataframe, df, on='date', how='left')
        return dataframe