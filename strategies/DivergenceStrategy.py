from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime
from freqtrade.persistence import Trade
from typing import Optional, Union

class DivergenceStrategy(IStrategy):

    minimal_roi = {

    }
    stoploss = -0.5
    timeframe = '1h'

    trailing_stop = True
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.5
    trailing_only_offset_is_reached = True

    can_short = True

    pivot_period = 5
    showlimit = 1
    maxpp = 10
    maxbars = 100
    dontconfirm = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=10)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        dataframe['stoch_k'], dataframe['stoch_d'] = ta.STOCH(dataframe)

        dataframe['VWAP'] = (dataframe['close'] * dataframe['volume']).cumsum() / dataframe['volume'].cumsum()
        ema_fast = dataframe['VWAP'].ewm(span=12, adjust=False).mean()
        ema_slow = dataframe['VWAP'].ewm(span=26, adjust=False).mean()
        dataframe['vwmacd'] = ema_fast - ema_slow
        dataframe['vwmacdsignal'] = dataframe['vwmacd'].ewm(span=9, adjust=False).mean()
        dataframe['vwmhist'] = dataframe['vwmacd'] - dataframe['vwmacdsignal']

        dataframe['cmf'] = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low']) * dataframe['volume']
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        return dataframe

    def divergence(self, series, pivot_low=True):
        """
        检查常规或隐藏背离
        """
        divergence_index = []
        for i in range(self.pivot_period, len(series) - self.pivot_period):
            if pivot_low:
                if series[i] < series[i - self.pivot_period] and series[i] < series[i + self.pivot_period]:
                    divergence_index.append(i)
            else:
                if series[i] > series[i - self.pivot_period] and series[i] > series[i + self.pivot_period]:
                    divergence_index.append(i)
        return divergence_index

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成买入信号
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0

        pivots = self.divergence(dataframe['rsi'], pivot_low=True)
        for pivot in pivots:
            if pivot + self.maxbars < len(dataframe):
                if dataframe['rsi'].iloc[pivot] > dataframe['rsi'].iloc[pivot + self.maxbars] and dataframe['close'].iloc[pivot] < dataframe['close'].iloc[pivot + self.maxbars]:
                    dataframe.loc[dataframe.index[pivot], 'enter_long'] = 1
                    dataframe.loc[dataframe.index[pivot], 'enter_tag'] = 'long'

        pivots = self.divergence(dataframe['rsi'], pivot_low=False)
        for pivot in pivots:
            if pivot + self.maxbars < len(dataframe):
                if dataframe['rsi'].iloc[pivot] < dataframe['rsi'].iloc[pivot + self.maxbars] and dataframe['close'].iloc[pivot] > dataframe['close'].iloc[pivot + self.maxbars]:
                    dataframe.loc[dataframe.index[pivot], 'enter_short'] = 1
                    dataframe.loc[dataframe.index[pivot], 'enter_tag'] = 'short'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成卖出信号
        """

        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        pivots = self.divergence(dataframe['rsi'], pivot_low=False)
        for pivot in pivots:
            if pivot + self.maxbars < len(dataframe):
                if dataframe['rsi'].iloc[pivot] < dataframe['rsi'].iloc[pivot + self.maxbars] and dataframe['close'].iloc[pivot] > dataframe['close'].iloc[pivot + self.maxbars]:
                    dataframe.loc[dataframe.index[pivot], 'exit_long'] = 1

        pivots = self.divergence(dataframe['rsi'], pivot_low=True)
        for pivot in pivots:
            if pivot + self.maxbars < len(dataframe):
                if dataframe['rsi'].iloc[pivot] > dataframe['rsi'].iloc[pivot + self.maxbars] and dataframe['close'].iloc[pivot] < dataframe['close'].iloc[pivot + self.maxbars]:
                    dataframe.loc[dataframe.index[pivot], 'exit_short'] = 1

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        open_trades = Trade.get_open_trades()

        num_shorts, num_longs = 0, 0
        for trade in open_trades:
            if "short" in trade.enter_tag:
                num_shorts += 1
            elif "long" in trade.enter_tag:
                num_longs += 1

        if side == "long" and num_longs >= 5:
            return False

        if side == "short" and num_shorts >= 5:
            return False

        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return 5
