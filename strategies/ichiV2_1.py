from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_open
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import technical.indicators as ftt
from functools import reduce

class ichiV2_1(IStrategy):

    minimal_roi = {
        "0": 0.05,
        "10": 0.03,
        "41": 0.01,
        "114": 0,
    }

    stoploss = -0.05

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        for interval in [5, 15, 30, 60, 120, 240, 360, 480]:
            dataframe[f'trend_close_{interval}m'] = ta.EMA(dataframe['close'], timeperiod=interval)
            dataframe[f'trend_open_{interval}m'] = ta.EMA(dataframe['open'], timeperiod=interval)

        dataframe['fan_magnitude'] = dataframe['trend_close_60m'] / dataframe['trend_close_480m']
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        dataframe['atr'] = ta.ATR(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['fan_magnitude_gain'] >= 1.002)
        conditions.append(dataframe['fan_magnitude'] > 1)

        for x in range(3):
            conditions.append(dataframe[f'fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        for interval in [5, 15, 30, 60, 120, 240, 360, 480]:
            conditions.append(dataframe[f'trend_close_{interval}m'] > dataframe[f'trend_open_{interval}m'])

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe['trend_close_120m']))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
