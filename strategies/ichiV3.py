
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
import numpy as np
from freqtrade.strategy import stoploss_from_open


class ichiV3(IStrategy):
    """
    This is the next version of ichiVx, and the previous version was ichiV2_5.

    ==============================
    Summary of changes from ichiV2_5:
    1. Does not alter the original OHLC data. The Heikin Ashi data is now only
       for calculating signals.
    2. The buy and sell parameters are the same as in the original ichiV1
       version.
    """
    INTERFACE_VERSION = 2


    buy_params = {
        "buy_trend_above_senkou_level": 5,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002


    }


    sell_params = {
        "sell_trend_indicator": "trend_close_2h",
    }





    stoploss = -0.06

    timeframe = '5m'

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.4
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    sell_profit_only = False


    plot_config = {
        'main_plot': {

            'senkou_a': {
                'color': 'green', #optional
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #optional
                'fill_color': 'rgba(255,76,46,0.2)', #optional
            },

            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'},
            'trend_open_5m': {'color': '#4D7BF5'},
            'trend_open_15m': {'color': '#1E8C38'},
            'trend_open_30m': {'color': '#ABA6D0'},
            'trend_open_1h': {'color': '#631B69'},
            'trend_open_2h': {'color': '#A553FF'},
            'trend_open_4h': {'color': '#B10C78'},
            'trend_open_6h': {'color': '#389E03'},
            'trend_open_8h': {'color': '#571E60'}
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']




        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=12)
        dataframe['trend_close_1.5h'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=18)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['trend_close_5m'], timeperiod=96)


        dataframe['trend_open_5m'] = dataframe['ha_open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['trend_open_5m'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['trend_open_5m'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['trend_open_5m'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['trend_open_5m'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['trend_open_5m'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['trend_open_5m'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['trend_open_5m'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['chikou_span'] = ichimoku['chikou_span']  # do not use this in live, it has lookahead bias
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe['atr'] = ta.ATR(dataframe)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        if self.buy_params['buy_trend_above_senkou_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_5m'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_above_senkou_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_15m'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_above_senkou_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_30m'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_above_senkou_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_1h'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_above_senkou_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_2h'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_above_senkou_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_4h'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_above_senkou_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_6h'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_above_senkou_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'].shift(1) > dataframe['senkou_a'].shift(1))
            conditions.append(dataframe['trend_close_8h'].shift(1) > dataframe['senkou_b'].shift(1))

        if self.buy_params['buy_trend_bullish_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'].shift(1) > dataframe['trend_open_5m'])

        if self.buy_params['buy_trend_bullish_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'].shift(1) > dataframe['trend_open_15m'])

        if self.buy_params['buy_trend_bullish_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'].shift(1) > dataframe['trend_open_30m'])

        if self.buy_params['buy_trend_bullish_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'].shift(1) > dataframe['trend_open_1h'])

        if self.buy_params['buy_trend_bullish_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'].shift(1) > dataframe['trend_open_2h'])

        if self.buy_params['buy_trend_bullish_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'].shift(1) > dataframe['trend_open_4h'])

        if self.buy_params['buy_trend_bullish_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'].shift(1) > dataframe['trend_open_6h'])

        if self.buy_params['buy_trend_bullish_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'].shift(1) > dataframe['trend_open_8h'])

        conditions.append(dataframe['fan_magnitude_gain'].shift(1) >= self.buy_params['buy_min_fan_magnitude_gain'])
        conditions.append(dataframe['fan_magnitude'].shift(1) > 1)

        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            conditions.append(dataframe['fan_magnitude'].shift(x+2) < dataframe['fan_magnitude'].shift(1))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(
            qtpylib.crossed_below(
                dataframe['trend_close_5m'].shift(1),
                dataframe[self.sell_params['sell_trend_indicator']]
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
