from datetime import datetime
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from functools import reduce

class DT_250102_01(IStrategy):
    """
    Freqtrade策略示例：多时间框架布林带策略。
    在4H, 2H, 5M, 1M K线到达布林带上下轨时触发买卖信号。
    """

    # ROI 和其他配置项...
    minimal_roi = {
        "0": 0.1
    }

    # 时间框架列表
    timeframe = '5m'  # 主要时间框架用于数据获取
    informative_timeframes = ['1m', '5m', '2h', '4h']

    # 订单类型设置...
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    stoploss = -0.25  # 最大允许亏损比例

    def informative_pairs(self):
        """
        定义附加的时间框架信息对。
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, tf) for pair in pairs for tf in self.informative_timeframes]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算并填充技术指标到数据框中。
        """
        for tf in self.informative_timeframes:
            if metadata['timeframe'] == tf:
                continue
            
            inf_tf = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf)
            
            # 计算布林带指标
            bollinger = ta.BBANDS(inf_tf['close'], timeperiod=20)
            inf_tf[f'{tf}_bb_lowerband'] = bollinger['lowerband']
            inf_tf[f'{tf}_bb_upperband'] = bollinger['upperband']

            # 合并到主dataframe
            dataframe = dataframe.join(inf_tf[[f'{tf}_bb_lowerband', f'{tf}_bb_upperband']].ffill().iloc[-1])

        # 主时间框架上的布林带指标
        bollinger_main = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['main_bb_lowerband'] = bollinger_main['lowerband']
        dataframe['main_bb_upperband'] = bollinger_main['upperband']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义进入趋势的条件。
        """
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # 检查所有时间框架是否同时触碰布林带下轨
        for tf in self.informative_timeframes:
            condition = (dataframe['close'] <= dataframe[f'{tf}_bb_lowerband'])
            conditions.append(condition)

        # 如果所有时间框架都满足，则生成买入信号
        combined_condition = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[combined_condition, 'enter_long'] = 1
        dataframe.loc[combined_condition, 'enter_tag'] += 'all_bbands_lower'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义退出趋势的条件。
        """
        conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        # 检查所有时间框架是否同时触碰布林带上轨
        for tf in self.informative_timeframes:
            condition = (dataframe['close'] >= dataframe[f'{tf}_bb_upperband'])
            conditions.append(condition)

        # 如果所有时间框架都满足，则生成卖出信号
        combined_condition = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[combined_condition, 'exit_long'] = 1
        dataframe.loc[combined_condition, 'exit_tag'] += 'all_bbands_upper'

        return dataframe