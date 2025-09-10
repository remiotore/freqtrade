import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from functools import reduce

class mq20241227(IStrategy):
    """
    改进的 5 分钟时间框架策略：
    - 结合 15 分钟时间框架的趋势信号。
    - 动态调整 RSI 和布林带参数，适应不同市场波动性。
    """

    # 策略基础设置
    timeframe = '5m'  # 主时间框架为 5 分钟
    informative_timeframe = '15m'  # 辅助时间框架为 15 分钟
    process_only_new_candles = True  # 仅处理新蜡烛
    startup_candle_count = 50  # 策略启动所需的最小蜡烛数量

    # 动态 ROI 设置
    minimal_roi = {
        "0": 0.01,  # 持仓时间为 0 分钟时，达到 1% 收益卖出
        "3": 0.015, # 持仓时间为 3 分钟时，达到 1.5% 收益卖出
        "10": 0     # 持仓时间超过 10 分钟时，不设目标收益
    }

    # 固定止损设置
    stoploss = -0.03  # 最大止损为 -3%

    def informative_pairs(self):
        """
        定义主时间框架和 15 分钟时间框架的币对。
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算策略所需的所有技术指标，并将结果添加到主时间框架 DataFrame。
        """
        # RSI 指标（默认周期为 14）
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # 动态布林带：调整标准差系数以适应市场波动性
        avg_volatility = dataframe['close'].rolling(20).std() / dataframe['close'].rolling(20).mean()
        dynamic_nbdev = 2.0 + avg_volatility.mean()  # 取平均值，确保为单一浮点数
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=dynamic_nbdev, nbdevdn=dynamic_nbdev)
        dataframe['boll_upper'] = bollinger['upperband']
        dataframe['boll_middle'] = bollinger['middleband']
        dataframe['boll_lower'] = bollinger['lowerband']

        # 均线
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)

        # MACD 指标
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # 成交量均值
        dataframe['volume_mean_10'] = dataframe['volume'].rolling(10).mean()

        return dataframe

    def populate_indicators_informative(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算 15 分钟时间框架的技术指标。
        """
        # EMA 和 MACD 指标
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)  # 长期趋势均线
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义买入信号的条件，包括震荡行情和趋势行情两种逻辑。
        """
        conditions = []

        # 获取 15 分钟时间框架数据
        informative = self.dp.get_pair_dataframe(metadata['pair'], self.informative_timeframe)

        # 确保 15 分钟时间框架指标已正确计算
        informative = self.populate_indicators_informative(informative, metadata)

        # 动态调整 RSI 超卖阈值：根据布林带宽度动态设置
        dataframe['rsi_dynamic'] = 30 + (dataframe['boll_upper'] - dataframe['boll_lower']).rolling(10).mean()

        # 震荡行情买入条件
        oscillation_condition = (
            (dataframe['rsi'] < dataframe['rsi_dynamic']) &  # 动态 RSI 超卖
            (dataframe['close'] < dataframe['boll_lower']) &  # 接近布林带下轨
            (dataframe['volume'] > dataframe['volume_mean_10'])  # 成交量放大
        )

        # 趋势行情买入条件（结合 15 分钟时间框架）
        trend_condition = (
            (dataframe['ema_5'] > dataframe['ema_20']) &  # 短期均线金叉长期均线
            (dataframe['macd'] > dataframe['macdsignal']) &  # MACD 金叉
            (informative['macd'] > informative['macdsignal'].shift(1)) &  # 15 分钟 MACD 金叉确认
            (informative['ema_50'] < dataframe['close']) &  # 确保价格在 15 分钟长期趋势均线上方
            (dataframe['volume'] > dataframe['volume_mean_10'] * 1.5)  # 成交量显著放大
        )

        # 将条件添加到列表
        conditions.append(oscillation_condition)
        conditions.append(trend_condition)

        # 应用条件，设置买入信号
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义卖出信号的条件。
        由 minimal_roi 自动处理，无需额外定义。
        """
        dataframe['exit_long'] = 0  # 卖出逻辑通过 ROI 实现
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        自定义止损逻辑：动态调整止损。
        """
        # 如果当前盈利超过 2%，设置更紧的止损以保护利润
        if current_profit > 0.02:
            return -0.01  # 止损设为 -1%
        # 如果当前亏损超过 2%，允许最大亏损为 -3%
        elif current_profit < -0.02:
            return -0.03
        # 默认止损为 -3%
        return -0.03
