from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class EmaRsiBounce(IStrategy):
    # === 核心参数 ===
    timeframe = '5m'  # 改为5分钟
    minimal_roi = {
        "0": 0.015,   # 1.5%快速止盈
        "10": 0.025,  # 10分钟后2.5%
        "30": 0.035,  # 30分钟后3.5%
        "60": 0.05,   # 1小时后5%
    }
    stoploss = -0.015  # -1.5%止损，更严格
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1%跟踪止损
    trailing_stop_positive_offset = 0.02  # 2%偏移
    trailing_only_offset_is_reached = True

    # 控制并发仓位
    max_open_trades = 5  # 增加并发数
    position_adjustment_enable = True  # 启用仓位调整

    # === 指标 ===
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 基础指标
        dataframe['ema12'] = ta.EMA(dataframe, timeperiod=12)  # 更短期的EMA
        dataframe['ema26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        
        # RSI指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=14)  # RSI均线
        
        # 布林带
        bb_lower, bb_middle, bb_upper = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bb_lower
        dataframe['bb_upper'] = bb_upper
        dataframe['bb_middle'] = bb_middle
        dataframe['bb_width'] = (bb_upper - bb_lower) / bb_middle  # 布林带宽度
        
        # MACD
        macd, macdsignal, macdhist = ta.MACD(dataframe['close'])
        dataframe['macd'] = macd
        dataframe['macdsignal'] = macdsignal
        dataframe['macdhist'] = macdhist
        
        # 成交量指标
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # 价格动量
        dataframe['price_change'] = dataframe['close'].pct_change()
        dataframe['price_change_ma'] = ta.SMA(dataframe['price_change'], timeperiod=10)
        
        # 波动率
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_ratio'] = dataframe['atr'] / dataframe['close']
        
        # 趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        return dataframe

    # === 买点 ===
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            # 趋势条件
            (dataframe['ema12'] > dataframe['ema26']) &
            (dataframe['ema26'] > dataframe['ema50']) &
            
            # 超跌反弹条件
            (dataframe['close'] < dataframe['bb_lower'] * 1.02) &  # 接近布林带下轨
            (dataframe['rsi'] < 30) &  # RSI超卖
            (dataframe['rsi'] > dataframe['rsi_ma']) &  # RSI开始回升
            
            # 成交量确认
            (dataframe['volume_ratio'] > 1.2) &  # 放量
            
            # 动量确认
            (dataframe['price_change_ma'] > 0) &  # 价格动量向上
            
            # 趋势强度
            (dataframe['adx'] > 20) &  # 趋势明确
            
            # 避免过度波动
            (dataframe['atr_ratio'] < 0.05),  # 波动率适中
            
            'buy'
        ] = 1
        
        # 强势突破买入
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_upper']) &
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['rsi'] > 50) &
            (dataframe['adx'] > 25),
            'buy'
        ] = 1
        
        return dataframe

    # === 卖点 ===
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            # 趋势反转
            (dataframe['ema12'] < dataframe['ema26']) &
            (dataframe['close'] < dataframe['ema50']),
            'sell'
        ] = 1
        
        # RSI超买
        dataframe.loc[
            (dataframe['rsi'] > 75) &
            (dataframe['close'] > dataframe['bb_upper']),
            'sell'
        ] = 1
        
        # MACD死叉
        dataframe.loc[
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macdhist'] < 0),
            'sell'
        ] = 1
        
        return dataframe

    # === 仓位管理 ===
    def custom_stake_amount(self, pair: str, current_time, current_rate,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: str, side: str,
                           **kwargs) -> float:
        """动态仓位管理"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 根据信号强度调整仓位
        if last_candle['volume_ratio'] > 2.0 and last_candle['rsi'] < 25:
            return max_stake * 0.8  # 强信号，大仓位
        elif last_candle['volume_ratio'] > 1.5 and last_candle['rsi'] < 30:
            return max_stake * 0.6  # 中等信号，中等仓位
        else:
            return max_stake * 0.4  # 弱信号，小仓位
