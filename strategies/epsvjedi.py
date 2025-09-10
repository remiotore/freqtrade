from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime

class epsvjedi(IStrategy):
    timeframe = '1m'
    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.005
    trailing_only_offset_is_reached = True
       
    sensitivity = 250
    risk_percent = 1
    break_even_target = "2"
    tp1_percent = 0.3
    tp1_percent_fix = 29
    tp2_percent = 0.6
    tp2_percent_fix = 25
    tp3_percent = 1.0
    tp3_percent_fix = 10
    tp4_percent = 2.0
    tp4_percent_fix = 0
    fixed_stop = False
    sl_percent = 0.15
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['highest'] = dataframe['high'].rolling(window=self.sensitivity).max()
        dataframe['lowest'] = dataframe['low'].rolling(window=self.sensitivity).min()
        dataframe['range'] = dataframe['highest'] - dataframe['lowest']
        dataframe['fib_236'] = dataframe['highest'] - dataframe['range'] * 0.236
        dataframe['fib_382'] = dataframe['highest'] - dataframe['range'] * 0.382
        dataframe['fib_5'] = dataframe['highest'] - dataframe['range'] * 0.5
        dataframe['fib_618'] = dataframe['highest'] - dataframe['range'] * 0.618
        dataframe['fib_786'] = dataframe['highest'] - dataframe['range'] * 0.786
        dataframe['imba_trend_line'] = dataframe['fib_5']

        dataframe['can_long'] = (dataframe['close'] >= dataframe['imba_trend_line']) & (dataframe['close'] >= dataframe['fib_236'])
        dataframe['prev_can_long'] = dataframe['can_long'].shift(1)
        dataframe['can_short'] = (dataframe['close'] <= dataframe['imba_trend_line']) & (dataframe['close'] <= dataframe['fib_786'])
        dataframe['prev_can_short'] = dataframe['can_short'].shift(1)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['can_long']) &
                (dataframe['prev_can_long'] == 0) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['close'], dataframe['fib_382'])) |
                (qtpylib.crossed_below(dataframe['close'], dataframe['fib_5'])) |
                (qtpylib.crossed_below(dataframe['close'], dataframe['fib_618'])) |
                (qtpylib.crossed_below(dataframe['close'], dataframe['fib_786']))
            ),
            'exit_long'] = 1
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        if self.fixed_stop:
            return -self.sl_percent
        else:
            return -0.15
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if current_profit >= self.tp1_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_236']).any():
                return 'tp1'
        elif current_profit >= self.tp2_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_382']).any():
                return 'tp2'
        elif current_profit >= self.tp3_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_618']).any():
                return 'tp3'
        elif current_profit >= self.tp4_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_786']).any():
                return 'tp4'
    
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if current_profit >= self.tp1_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_236']).any():
                return 'tp1'
        elif current_profit >= self.tp2_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_382']).any():
                return 'tp2'
        elif current_profit >= self.tp3_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_618']).any():
                return 'tp3'
        elif current_profit >= self.tp4_percent / 100:
            if qtpylib.crossed_above(dataframe['close'], dataframe['fib_786']).any():
                return 'tp4'
