
from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd
import numpy as np
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_minutes
import logging

logger = logging.getLogger(__name__)









def ssl_atr(dataframe, length = 7):
    df = dataframe.copy()
    df['smaHigh'] = df['high'].rolling(length).mean() + df['atr']
    df['smaLow'] = df['low'].rolling(length).mean() - df['atr']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

class Obelisk_Ichimoku_ZEMA_v1_555(IStrategy):

    timeframe = '5m'

    informative_timeframe = '1h'


    startup_candle_count = 200


    process_only_new_candles = True

    minimal_roi = {
        "0": 0.078,
        "40": 0.062,
        "99": 0.039,
        "218": 0
    }

    stoploss = -0.294

    buy_params = {
     'low_offset': 0.964, 'zema_len_buy': 51
    }

    sell_params = {
     'high_offset': 1.004, 'zema_len_sell': 72
    }

    low_offset = DecimalParameter(0.80, 1.20, default=1.004, space='buy', optimize=True)
    high_offset = DecimalParameter(0.80, 1.20, default=0.964, space='sell', optimize=True)
    zema_len_buy = IntParameter(30, 90, default=72, space='buy', optimize=True)
    zema_len_sell = IntParameter(30, 90, default=51, space='sell', optimize=True)

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
        dataframe['future_red'] = (dataframe['leading_senkou_span_a'] < dataframe['leading_senkou_span_b']).astype('int') * 2



        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['cloud_top'])
            ).shift(displacement).fillna(0).astype('int')

        dataframe['chikou_low'] = (
                (dataframe['chikou_span'] < dataframe['cloud_bottom'])
            ).shift(displacement).fillna(0).astype('int')


        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        ssl_down, ssl_up = ssl_atr(dataframe, 10)
        dataframe['ssl_down'] = ssl_down
        dataframe['ssl_up'] = ssl_up
        dataframe['ssl_ok'] = (
                (ssl_up > ssl_down) 
            ).astype('int') * 3
        dataframe['ssl_bear'] = (
                (ssl_up < ssl_down) 
            ).astype('int') * 3

        dataframe['ichimoku_ok'] = (
                (dataframe['tenkan_sen'] > dataframe['kijun_sen'])
                & (dataframe['close'] > dataframe['cloud_top'])
                & (dataframe['future_green'] > 0) 

            ).astype('int') * 4

        dataframe['ichimoku_bear'] = (
                (dataframe['tenkan_sen'] < dataframe['kijun_sen'])
                & (dataframe['close'] < dataframe['cloud_bottom'])
                & (dataframe['future_red'] > 0) 

            ).astype('int') * 4

        dataframe['ichimoku_valid'] = (
                (dataframe['leading_senkou_span_b'] == dataframe['leading_senkou_span_b']) # not NaN
            ).astype('int') * 1

        dataframe['trend_pulse'] = (
                (dataframe['ichimoku_ok'] > 0) 
                & (dataframe['ssl_ok'] > 0)
            ).astype('int') * 2

        dataframe['bear_trend_pulse'] = (
                (dataframe['ichimoku_bear'] > 0) 
                & (dataframe['ssl_bear'] > 0)
            ).astype('int') * 2


        dataframe['trend_over'] = (
                (dataframe['ssl_ok'] == 0)
                | (dataframe['close'] < dataframe['cloud_top'])
            ).astype('int') * 1

        dataframe['bear_trend_over'] = (
                (dataframe['ssl_bear'] == 0)
                | (dataframe['close'] > dataframe['cloud_bottom'])
            ).astype('int') * 1

        dataframe.loc[ (dataframe['trend_pulse'] > 0), 'trending'] = 3
        dataframe.loc[ (dataframe['trend_over'] > 0) , 'trending'] = 0
        dataframe['trending'].fillna(method='ffill', inplace=True)

        dataframe.loc[ (dataframe['bear_trend_pulse'] > 0), 'bear_trending'] = 3
        dataframe.loc[ (dataframe['bear_trend_over'] > 0) , 'bear_trending'] = 0
        dataframe['bear_trending'].fillna(method='ffill', inplace=True)

        return dataframe

    def fast_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            for len in range(30, 91):
                dataframe[f'zema_{len}'] = ftt.zema(dataframe, period=len)
        else:
            dataframe[f'zema_{self.zema_len_buy.value}'] = ftt.zema(dataframe, period=self.zema_len_buy.value)
            dataframe[f'zema_{self.zema_len_sell.value}'] = ftt.zema(dataframe, period=self.zema_len_sell.value)
            dataframe[f'zema_buy'] = ftt.zema(dataframe, period=self.zema_len_buy.value) * self.low_offset.value
            dataframe[f'zema_sell'] = ftt.zema(dataframe, period=self.zema_len_sell.value) * self.high_offset.value


        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert (timeframe_to_minutes(self.timeframe) == 5), "Run this strategy at 5m."

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
        zema = f'zema_{self.zema_len_buy.value}'

        dataframe.loc[
            (dataframe['ichimoku_valid'] > 0)
            & (dataframe['bear_trending'] == 0)
            & (qtpylib.crossed_above(dataframe['close'], (dataframe[zema] * self.low_offset.value)))

        , 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        zema = f'zema_{self.zema_len_sell.value}'

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], (dataframe[zema] * self.high_offset.value))
            )
        , 'sell'] = 1

        return dataframe

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: 'datetime', **kwargs) -> bool:

        if sell_reason in ('roi',):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = dataframe.iloc[-1]
            if current_candle is not None:
                current_candle = current_candle.squeeze()

                if current_candle['trending'] > 0:
                    return False

        return True

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


            'ssl_up': { 'color': 'green' },




            'zema_buy': { 'color': 'blue' },
            'zema_sell': { 'color': 'orange' },
        },
        'subplots': {
            "Trend": {
                'trending': {'color': 'green'},
                'bear_trending': {'color': 'red'},
            },
            "Bull": {
                'trend_pulse': {'color': 'blue'},
                'trending': {'color': 'orange'},
                'trend_over': {'color': 'red'},
            },
            "Bull Signals": {
                'ichimoku_ok': {'color': 'green'},
                'ssl_ok': {'color': 'red'},
            },
            "Bear": {
                'bear_trend_pulse': {'color': 'blue'},
                'bear_trending': {'color': 'orange'},
                'bear_trend_over': {'color': 'red'},
            },
            "Bear Signals": {
                'ichimoku_bear': {'color': 'green'},
                'ssl_bear': {'color': 'red'},
            },
            "Misc": {
                'ichimoku_valid': {'color': 'green'},
            },
        }
    }
