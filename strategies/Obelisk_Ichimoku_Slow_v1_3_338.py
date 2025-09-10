
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd
import numpy as np
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes















































def ssl_atr(dataframe, length = 7):
    df = dataframe.copy()
    df['smaHigh'] = df['high'].rolling(length).mean() + df['atr']
    df['smaLow'] = df['low'].rolling(length).mean() - df['atr']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

class Obelisk_Ichimoku_Slow_v1_3_338(IStrategy):

    timeframe = '1h'

    informative_timeframe = '1h'




    startup_candle_count = 180


    process_only_new_candles = True

    minimal_roi = {
        "0": 0.10,
        "60": 0.072,
        "120": 0.049,
        "240": 0.02,
        "360": 0,
    }

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.04




    stoploss = -0.99


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def slow_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        displacement = 30
        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=displacement
            )

        dataframe['chikou_span'] = ichimoku['chikou_span']

        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        dataframe.loc[:, 'cloud_top'] = dataframe.loc[:, ['senkou_a', 'senkou_b']].max(axis=1)
        dataframe.loc[:, 'cloud_bottom'] = dataframe.loc[:, ['senkou_a', 'senkou_b']].min(axis=1)


        dataframe['future_green'] = (dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']).astype('int') * 2



        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['senkou_a']) &
                (dataframe['chikou_span'] > dataframe['senkou_b'])
            ).shift(displacement).fillna(0).astype('int')


        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_ok'] = (
                (dataframe['close'] > dataframe['ema50'])
                & (dataframe['ema50'] > dataframe['ema200'])
            ).astype('int') * 2

        dataframe['efi_base'] = ((dataframe['close'] - dataframe['close'].shift()) * dataframe['volume'])
        dataframe['efi'] = ta.EMA(dataframe['efi_base'], 13)
        dataframe['efi_ok'] = (dataframe['efi'] > 0).astype('int')

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        ssl_down, ssl_up = ssl_atr(dataframe, 10)
        dataframe['ssl_down'] = ssl_down
        dataframe['ssl_up'] = ssl_up
        dataframe['ssl_ok'] = (
                (ssl_up > ssl_down) 
            ).astype('int') * 3

        dataframe['ichimoku_ok'] = (
                (dataframe['tenkan_sen'] > dataframe['kijun_sen'])
                & (dataframe['close'] > dataframe['cloud_top'])
                & (dataframe['future_green'] > 0) 
                & (dataframe['chikou_high'] > 0) 
            ).astype('int') * 4

        dataframe['entry_ok'] = (
                (dataframe['efi_ok'] > 0)
                & (dataframe['open'] < dataframe['ssl_up'])
                & (dataframe['close'] < dataframe['ssl_up'])
            ).astype('int') * 1

        dataframe['trend_pulse'] = (
                (dataframe['ichimoku_ok'] > 0) 
                & (dataframe['ssl_ok'] > 0)
                & (dataframe['ema_ok'] > 0)
            ).astype('int') * 2

        dataframe['trend_over'] = (
                (dataframe['ssl_ok'] == 0)
            ).astype('int') * 1

        dataframe.loc[ (dataframe['trend_pulse'] > 0), 'trending'] = 3
        dataframe.loc[ (dataframe['trend_over'] > 0) , 'trending'] = 0
        dataframe['trending'].fillna(method='ffill', inplace=True)

        return dataframe

    def fast_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:




        if self.timeframe == self.informative_timeframe:
            dataframe = self.slow_tf_indicators(dataframe, metadata)
        else:
            assert self.dp, "DataProvider is required for multiple timeframes."

            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
            informative = self.slow_tf_indicators(informative.copy(), metadata)

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

            skip_columns = [(s + "_" + self.informative_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s, inplace=True)

        dataframe = self.fast_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['trending'] > 0)
            & (dataframe['entry_ok'] > 0)
            & (dataframe['date'].dt.minute == 0) # when backtesting at 5m/1m only set signal on the hour
        , 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['trending'] == 0)
            & (dataframe['date'].dt.minute == 0) # when backtesting at 5m/1m only set signal on the hour
            , 'sell'] = 1
        return dataframe

    plot_config = {

        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(0,0,0,0.2)',
            },

            'senkou_b': {
                'color': 'red',
            },
            'tenkan_sen': { 'color': 'blue' },
            'kijun_sen': { 'color': 'orange' },





        },
        'subplots': {
            "Trend": {
                'trend_pulse': {'color': 'blue'},
                'trending': {'color': 'orange'},
                'trend_over': {'color': 'red'},
            },
            "Signals": {
                'ichimoku_ok': {'color': 'green'},
                'ssl_ok': {'color': 'red'},
                'ema_ok': {'color': 'orange'},
                'entry_ok': {'color': 'blue'},
            },
        }
    }
