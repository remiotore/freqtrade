# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from typing import Optional
from datetime import datetime, timedelta
import numpy as np
from freqtrade.strategy import CategoricalParameter, IntParameter,RealParameter
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

class AlvinchenV1(IStrategy):

    # NOTE: settings as of the 25th july 21
    # Buy hyperspace params:
    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002 # NOTE: Good value (Win% ~70%), alot of trades
        #"buy_min_fan_magnitude_gain": 1.008 # NOTE: Very save value (Win% ~90%), only the biggest moves 1.008,
    }

    # Sell hyperspace params:
    # NOTE: was 15m but kept bailing out in dryrun
    sell_params = {
        "sell_trend_indicator": "trend_close_2h",
    }

    # ROI table:
    minimal_roi = {
        "0": 0.059,
        "10": 0.037,
        "41": 0.012,
        "114": 0
    }

    # Stoploss:
    stoploss = -0.275

    # Optimal timeframe for the strategy
    timeframe = '5m'

    startup_candle_count = 240
    process_only_new_candles = False

    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = True
    position_adjustment_enable = True

    plot_config = {
        'main_plot': {
            # fill area between senkou_a and senkou_b
            'senkou_a': {
                'color': 'green', #optional
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #optional
                'fill_color': 'rgba(255,76,46,0.2)', #optional
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'}
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

    # 定义可以进行 hyperopt 的参数
    buy_trend_above_senkou_level = IntParameter(1, 8, default=1, optimize=True)
    buy_trend_bullish_level = IntParameter(1, 8, default=6, optimize=True)
    buy_fan_magnitude_shift_value = IntParameter(1, 5, default=3, optimize=True)
    buy_min_fan_magnitude_gain = RealParameter(1.001, 1.01, default=1.002, optimize=True)
    sell_trend_indicator = CategoricalParameter(["trend_close_5m", "trend_close_15m", "trend_close_30m", "trend_close_1h", "trend_close_2h", "trend_close_4h", "trend_close_6h", "trend_close_8h"], default="trend_close_2h", optimize=True)

    # 定义保护机制的参数
    cooldown_period = CategoricalParameter([5, 10, 15], default=5, optimize=True)
    max_drawdown_lookback = CategoricalParameter([120, 150, 180], default=120, optimize=True)
    max_drawdown_trade_limit = CategoricalParameter([0.2, 0.25, 0.3], default=0.2, optimize=True)
    max_drawdown_stop_duration = CategoricalParameter([60, 70, 80], default=60, optimize=True)
    stoploss_guard_lookback = CategoricalParameter([60, 70, 80], default=60, optimize=True)
    stoploss_guard_trade_limit = CategoricalParameter([0.1, 0.12, 0.15], default=0.1, optimize=True)
    stoploss_guard_stop_duration = CategoricalParameter([30, 35, 40], default=30, optimize=True)

    # 添加保护机制
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_period.value  # 交易后冷却 5 个蜡烛周期
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,  # 回看 120 个蜡烛周期
                "trade_limit": self.max_drawdown_trade_limit.value,  # 最大回撤 20%
                "stop_duration_candles": self.max_drawdown_stop_duration.value  # 触发后停止 60 个蜡烛周期
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,  # 回看 60 个蜡烛周期
                "trade_limit": self.stoploss_guard_trade_limit.value,  # 最大止损 10%
                "stop_duration_candles": self.stoploss_guard_stop_duration.value  # 触发后停止 30 个蜡烛周期
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        #dataframe['close'] = heikinashi['close']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96)

        dataframe['trend_open_5m'] = dataframe['open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

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

        dataframe['atr'] = ta.ATR(dataframe)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # Trending market
        if self.buy_trend_above_senkou_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])

        # Trends bullish
        if self.buy_trend_bullish_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])

        if self.buy_trend_bullish_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])

        if self.buy_trend_bullish_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])

        if self.buy_trend_bullish_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])

        if self.buy_trend_bullish_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])

        if self.buy_trend_bullish_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])

        if self.buy_trend_bullish_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])

        if self.buy_trend_bullish_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])

        # Trends magnitude
        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_min_fan_magnitude_gain.value)
        conditions.append(dataframe['fan_magnitude'] > 1)

        for x in range(self.buy_fan_magnitude_shift_value.value):
            conditions.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_trend_indicator.value]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
    
    
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
            current_rate: float, current_profit: float,
            min_stake: Optional[float], max_stake: float,
            current_entry_rate: float, current_exit_rate: float,
            current_entry_profit: float, current_exit_profit: float,
            **kwargs) -> Optional[float]:
            """
            自定义交易调整逻辑，返回交易应增加或减少的仓位金额。
            这意味着额外的买卖订单会产生额外的费用。
            仅在 `position_adjustment_enable` 设置为 True 时调用。

            参数:
            - trade (Trade): 当前交易对象。
            - current_time (datetime): 当前时间。
            - current_rate (float): 资产的当前价格。
            - current_profit (float): 交易的当前利润。
            - min_stake (Optional[float]): 最小仓位金额。
            - max_stake (float): 最大仓位金额。
            - current_entry_rate (float): 交易的当前入场价格。
            - current_exit_rate (float): 交易的当前出场价格。
            - current_entry_profit (float): 交易的当前入场利润。
            - current_exit_profit (float): 交易的当前出场利润。
            - **kwargs: 其他关键字参数。

            返回:
            - Optional[float]: 调整交易的仓位金额，如果不需要调整则返回 None。
            """

            # 获取交易对的 DataFrame（仅用于展示如何访问它）
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

            # 计算当前价格与入场价格的百分比变化
            price_change_percent = (current_rate - trade.open_rate) / trade.open_rate * 100

            # 根据价格变化和当前利润来调整仓位
            if price_change_percent > 10 and current_profit > 0.10 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)  # 价格上涨且利润较高时，减少一半仓位
            elif price_change_percent < -5 and current_profit < -0.05:
                return trade.stake_amount * 0.5  # 价格下跌且利润较低时，增加一半仓位
            elif current_profit > -0.04 and trade.nr_of_successful_entries == 1:
                return None  # 利润在一定范围内且只有一次入场时，不调整仓位
            elif current_profit > -0.06 and trade.nr_of_successful_entries == 2:
                return None  # 利润在一定范围内且有两次入场时，不调整仓位
            elif current_profit > -0.08 and trade.nr_of_successful_entries == 3:
                return None  # 利润在一定范围内且有三次入场时，不调整仓位

            filled_entries = trade.select_filled_orders(trade.entry_side)
            count_of_entries = trade.nr_of_successful_entries

            try:
                stake_amount = filled_entries[0].cost

                if count_of_entries > 1:
                    stake_amount = stake_amount * (1 + (count_of_entries - 1) * 0.5)  # 每增加一次入场，仓位增加50%

                return stake_amount

            except Exception as exception:
                logger.error(f"调整交易对 {trade.pair} 的 DCA 仓位时出错: {exception}")
                return None