from datetime import datetime
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
from freqtrade.strategy import IntParameter, DecimalParameter
from functools import reduce
import warnings
import numpy as np  # 确保引入 numpy

warnings.simplefilter(action="ignore", category=RuntimeWarning)


class maqun888(IStrategy):
    minimal_roi = {
        "0": 1
    }
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 240

    # Hyperopt Parameters
    buy_rsi_fast_32 = IntParameter(20.0, 70.0, default=62, space='buy', optimize=True)
    buy_rsi_32 = IntParameter(15.0, 50.0, default=19, space='buy', optimize=True)
    buy_sma15_32 = DecimalParameter(0.9, 1.0, default=0.97, space='buy', optimize=True)
    buy_cti_32 = DecimalParameter(-1.0, 1.0, default=0.89, space='buy', optimize=True)

    sell_fastx = IntParameter(50.0, 100.0, default=73, space='sell', optimize=True)
    sell_cci = IntParameter(0.0, 600.0, default=154, space='sell', optimize=True)

    trailing_stop = True
    trailing_stop_positive = 0.0008  # 跌幅超过 0.08% 触发卖出
    trailing_stop_positive_offset = 0.005  # 盈利达到 0.5% 开始追踪
    trailing_only_offset_is_reached = True

    stoploss = -1  # 禁用止损，通过设置极小值满足配置要求

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 添加技术指标
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_6'] = ta.RSI(dataframe, timeperiod=6)
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['macd_hist'] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)['macdhist']
        dataframe['macd'] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)['macd']
        dataframe['macdsignal'] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)['macdsignal']
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        boll = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['boll_lower'] = boll['lowerband']
        dataframe['boll_mid'] = boll['middleband']
        dataframe['sar_signal'] = ta.SAR(dataframe) < dataframe['close']  # SAR 上涨信号

        # 新增: ADX 强趋势
        dataframe['adx_strong'] = dataframe['adx'] > 30

        # 新增: CCI 超卖
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=14)
        dataframe['cci_oversold'] = dataframe['cci'] < -100

        # 新增: 突破布林带中轨
        dataframe['boll_mid_break'] = (
            (dataframe['close'] > dataframe['boll_mid']) & 
            (dataframe['close'].shift(1) < dataframe['boll_mid'])
        )

        # 新增: RSI 背离
        dataframe['rsi_divergence'] = (
            (dataframe['close'] < dataframe['close'].rolling(window=5).min()) &
            (dataframe['rsi'] > dataframe['rsi'].rolling(window=5).min())
        )

        # 新增: 支撑和阻力位
        dataframe['support'] = dataframe['low'].rolling(window=10).min()
        dataframe['resistance'] = dataframe['high'].rolling(window=10).max()

        # 计算 KDJ，如果计算失败，则使用 NaN 或默认值
        try:
            kdj = pta.kdj(dataframe)
            dataframe['kdj'] = kdj['J'] if kdj is not None else np.nan
        except Exception as e:
            dataframe['kdj'] = np.nan  # 如果发生异常，设置为 NaN

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long = []
        dataframe.loc[:, 'enter_tag'] = ''

        # 条件 1: RSI 6 小于 40 放宽至 40
        rsi6_condition = dataframe['rsi_6'] < 40
        conditions_long.append(rsi6_condition)

        # 条件 2: BOLL 下轨放量
        boll_condition = (
            (dataframe['close'] < dataframe['boll_lower']) & 
            (dataframe['volume'] > dataframe['volume'].rolling(window=10).mean()) &  # 放宽为 10
            (dataframe['close'] > dataframe['open'])
        )
        conditions_long.append(boll_condition)

        # 条件 3: RSI 14 小于 30 放宽至 35
        rsi14_condition = dataframe['rsi'] < 35
        conditions_long.append(rsi14_condition)

        # 条件 4: 收盘站上 EMA5 和 MA5 且放量
        ema_ma_condition = (
            (dataframe['close'] > dataframe['ema_5']) & 
            (dataframe['close'] > dataframe['ma_5']) & 
            (dataframe['volume'] > dataframe['volume'].rolling(window=10).mean()) &  # 放宽为 10
            (dataframe['close'] > dataframe['open'])
        )
        conditions_long.append(ema_ma_condition)

        # 条件 5: 连续1个SAR上涨信号且BOLL中轨以下 放宽为 1
        sar_boll_condition = (
            (dataframe['sar_signal']) & 
            (dataframe['close'] < dataframe['boll_mid'])
        )
        conditions_long.append(sar_boll_condition)

        # 条件 6: MACD 空心柱
        macd_hollow_condition = (dataframe['macd_hist'] > 0) & (dataframe['macd_hist'].shift(1) <= 0)
        conditions_long.append(macd_hollow_condition)

        # 条件 7: ADX 强趋势
        adx_strong_condition = dataframe['adx_strong']
        conditions_long.append(adx_strong_condition)

        # 条件 8: CCI 超卖
        cci_oversold_condition = dataframe['cci_oversold']
        conditions_long.append(cci_oversold_condition)

        # 条件 9: 突破布林带中轨
        boll_mid_break_condition = dataframe['boll_mid_break']
        conditions_long.append(boll_mid_break_condition)

        # 条件 10: RSI 背离
        rsi_divergence_condition = dataframe['rsi_divergence']
        conditions_long.append(rsi_divergence_condition)

        # 条件 11: MACD 信号线交叉
        macd_signal_cross_condition = (dataframe['macd'] > dataframe['macdsignal']) & (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
        conditions_long.append(macd_signal_cross_condition)

        # 条件 12: 价格突破前5根K线的高点
        dataframe['high_5'] = dataframe['high'].rolling(window=5).max()
        breakout_condition = dataframe['close'] > dataframe['high_5']
        conditions_long.append(breakout_condition)

        # 条件 13: 支撑位突破
        support_breakout_condition = dataframe['close'] > dataframe['resistance']
        conditions_long.append(support_breakout_condition)

        # 条件 14: 价格突破EMA20
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        price_ema_condition = dataframe['close'] > dataframe['ema_20']
        conditions_long.append(price_ema_condition)

        # 条件 15: 成交量突破5日均值
        volume_increase_condition = dataframe['volume'] > dataframe['volume'].rolling(window=5).mean() * 1.5
        conditions_long.append(volume_increase_condition)

        # 条件 16: 均线斜率为正
        dataframe['sma_15_slope'] = dataframe['sma_15'] - dataframe['sma_15'].shift(5)  # 5 根 K 线的变化量
        sma_slope_condition = dataframe['sma_15_slope'] > 0
        conditions_long.append(sma_slope_condition)

        # 条件 17: 价格回调至支撑位
        price_pullback_to_support_condition = dataframe['close'] > dataframe['support'] * 0.98
        conditions_long.append(price_pullback_to_support_condition)

        # 条件 18: 上涨趋势中的回调
        pullback_in_uptrend_condition = dataframe['close'] > dataframe['ema_5']  # 价格位于 EMA 5 上方
        conditions_long.append(pullback_in_uptrend_condition)

        # 条件 19: 价格突破过去100根K线的高点
        dataframe['high_100'] = dataframe['high'].rolling(window=100).max()
        high_100_breakout_condition = dataframe['close'] > dataframe['high_100']
        conditions_long.append(high_100_breakout_condition)

        # 条件 20: 对数收益率为正
        dataframe['log_returns'] = np.log(dataframe['close'] / dataframe['close'].shift(1))
        log_returns_condition = dataframe['log_returns'] > 0
        conditions_long.append(log_returns_condition)

        # 条件 21: 短期 EMA 5 穿越长期 EMA 20
        price_ema_cross_condition = dataframe['ema_5'] > dataframe['ema_20']
        conditions_long.append(price_ema_cross_condition)

        # 将所有条件组合
        dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 确保 dataframe.index 是 DatetimeIndex 类型
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            dataframe.index = pd.to_datetime(dataframe.index)

        conditions_exit_long = []

        # 条件 1: 持仓 6 小时且无亏损卖出
        six_hour_condition = (
            (dataframe['close'] >= dataframe['open']) &  # 无亏损
            (dataframe.index - dataframe.index[0] >= pd.Timedelta(hours=6))  # 持仓时间大于 6 小时
        )
        conditions_exit_long.append(six_hour_condition)

        # 条件 2: 持仓 12 小时且亏损 3% 卖出
        twelve_hour_condition = (
            (dataframe['close'] < dataframe['open'] * 0.97) &  # 亏损 3%
            (dataframe.index - dataframe.index[0] >= pd.Timedelta(hours=12))  # 持仓时间大于 12 小时
        )
        conditions_exit_long.append(twelve_hour_condition)

        # 条件 3: 持仓 13 小时且亏损 4% 卖出
        thirteen_hour_condition = (
            (dataframe['close'] < dataframe['open'] * 0.96) &  # 亏损 4%
            (dataframe.index - dataframe.index[0] >= pd.Timedelta(hours=13))  # 持仓时间大于 13 小时
        )
        conditions_exit_long.append(thirteen_hour_condition)

        # 条件 4: 持仓 24 小时且亏损 7% 卖出
        twenty_four_hour_condition = (
            (dataframe['close'] < dataframe['open'] * 0.93) &  # 亏损 7%
            (dataframe.index - dataframe.index[0] >= pd.Timedelta(hours=24))  # 持仓时间大于 24 小时
        )
        conditions_exit_long.append(twenty_four_hour_condition)

        # 将所有条件组合
        dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1

        return dataframe
