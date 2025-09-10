from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame, Series, DatetimeIndex, merge

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from datetime import datetime, timedelta









class CHTP_vtest(IStrategy):
    minimal_roi = {
        "0": 0.035,
        "60": 0.01,
    }
    
    stoploss = -0.2
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    startup_candle_count = 200
    process_only_new_candles = True

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 4
        }
    ]

    def informative_pairs(self):
        informative_pairs = [("BTC/USDT", "5m"),]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if not self.dp:

            return dataframe
        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe("BTC/USDT", "5m")
        informative['plus_di'] = ta.PLUS_DI(informative)
        informative['minus_di'] = ta.MINUS_DI(informative)
        informative['cmf'] = self.chaikin_mf(informative)
        informative['sma50'] = ta.SMA(informative, timeperiod=50)
        informative['sma200'] = ta.SMA(informative, timeperiod=200)
        informative['pct'] = informative['close'].pct_change(1)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['btc_bear'] = dataframe['sma200_5m'].lt(dataframe['sma200_5m'].shift(10))
        dataframe['btc_sma_delta'] = ( ( (dataframe['sma50_5m'] - dataframe['sma200_5m']) / dataframe['sma50_5m']) * 100)
        dataframe['pct'] = dataframe['close'].pct_change(1)
        dataframe['body_pct'] = ((dataframe['open'] - dataframe['close']) * 100) / ((dataframe['open'] - dataframe['close']) + (dataframe['close'] - dataframe['low']))

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            
            (
                (dataframe['btc_sma_delta'] > -2) &
                    (
                        (dataframe['cmf_5m'] < -0.25) &
                        (dataframe['minus_di_5m'] > dataframe['plus_di_5m']) &
                        (dataframe['close_5m'] < dataframe['sma50_5m']) &
                            (
                                (dataframe['btc_bear']) &
                                    (
                                        (dataframe['rsi'] < 30) &
                                        (dataframe['pct'] > -0.05) &
                                            (
                                                (dataframe['close'] <= 0.975 * dataframe['bb_lowerband']) &
                                                (dataframe['body_pct'] > 70)
                                            ) |
                                        (dataframe['pct'] < -0.05) &
                                            (
                                                (dataframe['close'] <= 0.70 * dataframe['bb_lowerband'])
                                            ) &
                                        (dataframe['close'] < dataframe['close'].shift(1)) &
                                        (dataframe['volume'] > 0)
                                    )
                            )
                    )

            ) ,

            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'sell'] = 0
        return dataframe

    def chaikin_mf(self, df, periods=20):
        close = df['close']
        low = df['low']
        high = df['high']
        volume = df['volume']

        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.fillna(0.0)
        mfv *= volume
        cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()

        return Series(cmf, name='cmf')