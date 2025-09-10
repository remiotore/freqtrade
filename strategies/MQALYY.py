# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce


class MQALYY(IStrategy):
    # 多头参数
    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002
    }

    # 空头参数
    short_params = {
        "short_trend_below_senkou_level": 1,
        "short_trend_bearish_level": 6,
        "short_fan_magnitude_shift_value": 3,
        "short_max_fan_magnitude_gain": 0.998
    }

    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.11

    # Optimal timeframe for the strategy
    timeframe = '5m'
    startup_candle_count = 96  # 保证足够的 K 线数量用于计算 24 小时数据
    process_only_new_candles = False

    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True  # 支持做空

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_4h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']

        # Highest price for dynamic take profit (long)
        dataframe['highest_close'] = dataframe['close'].cummax()
        # Lowest price for dynamic take profit (short)
        dataframe['lowest_close'] = dataframe['close'].cummin()

        # 6 小时、12 小时和 24 小时的最大价格
        dataframe['max_6h'] = dataframe['close'].rolling(window=72).max()
        dataframe['max_12h'] = dataframe['close'].rolling(window=144).max()
        dataframe['max_24h'] = dataframe['close'].rolling(window=288).max()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头开仓条件
        conditions_long = []
        conditions_long.append(dataframe['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain'])
        conditions_long.append(dataframe['fan_magnitude'] > 1)

        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            conditions_long.append(dataframe['fan_magnitude'].shift(x + 1) < dataframe['fan_magnitude'])

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'enter_long'] = 1

        # 空头开仓条件
        conditions_short = []
        conditions_short.append(dataframe['fan_magnitude_gain'] <= self.short_params['short_max_fan_magnitude_gain'])
        conditions_short.append(dataframe['fan_magnitude'] < 1)

        for x in range(self.short_params['short_fan_magnitude_shift_value']):
            conditions_short.append(dataframe['fan_magnitude'].shift(x + 1) > dataframe['fan_magnitude'])

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_short),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 动态止盈和追踪止损参数
        take_profit_threshold = 1.02  # 涨幅大于 2% 启动追踪止损
        trailing_stop_loss = 0.99     # 回撤超过 1% 卖出

        # 多头动态止盈和追踪止损条件
        conditions_long_tp = [
            dataframe['close'] >= dataframe['highest_close'] * take_profit_threshold
        ]
        conditions_long_trailing = [
            dataframe['close'] < dataframe['highest_close'] * trailing_stop_loss
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_long_tp) | 
            reduce(lambda x, y: x & y, conditions_long_trailing),
            'exit_long'
        ] = 1

        # 空头动态止盈和追踪止损条件
        conditions_short_tp = [
            dataframe['close'] <= dataframe['lowest_close'] / take_profit_threshold
        ]
        conditions_short_trailing = [
            dataframe['close'] > dataframe['lowest_close'] / trailing_stop_loss
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_short_tp) | 
            reduce(lambda x, y: x & y, conditions_short_trailing),
            'exit_short'
        ] = 1

        # 6 小时亏损大于 3%
        conditions_loss_6h = [
            dataframe['close'] <= dataframe['max_6h'] * 0.97
        ]

        # 12 小时亏损大于 4%
        conditions_loss_12h = [
            dataframe['close'] <= dataframe['max_12h'] * 0.96
        ]

        # 24 小时亏损大于 10%
        conditions_loss_24h = [
            dataframe['close'] <= dataframe['max_24h'] * 0.90
        ]

        # 将亏损条件应用到多头和空头退出
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_loss_6h) |
            reduce(lambda x, y: x & y, conditions_loss_12h) |
            reduce(lambda x, y: x & y, conditions_loss_24h),
            ['exit_long', 'exit_short']
        ] = 1

        return dataframe
