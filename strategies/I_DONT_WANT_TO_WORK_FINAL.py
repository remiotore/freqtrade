



from pandas import DataFrame
from typing import Optional
import pandas_ta as pta
from datetime import datetime
from freqtrade.strategy import (IStrategy, stoploss_from_absolute)
from freqtrade.persistence import Trade
from pandas_ta.statistics import stdev
import math



class I_DONT_WANT_TO_WORK_FINAL(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = True




    stoploss = -0.786

    trailing_stop = False


    timeframe = '30m'

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    m_trades = 12


    startup_candle_count: int = 1900

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    position_adjustment_enable = True
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = dataframe.iloc[-1].squeeze()
        side = 1 if trade.is_short else -1
        if trade.nr_of_successful_exits > 1:
            return stoploss_from_absolute(candle['VWMA'], 
                                        current_rate, is_short=trade.is_short,
                                        leverage=trade.leverage)
        if trade.nr_of_successful_exits == 1:
            return -current_profit
        if trade.nr_of_successful_exits == 0:
            return stoploss_from_absolute(candle['VWMA'] + (side * candle['deviation1']), 
                                        current_rate, is_short=trade.is_short,
                                        leverage=trade.leverage)



    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        return self.wallets.get_total_stake_amount() / self.m_trades

    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:        

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        long = (last_candle['close'] > last_candle['upper1'])
        short = (last_candle['close'] <= last_candle['lower1'])
        if short:
            upper2 = last_candle['upper2']
            low = last_candle['close']
            calc = abs((low-upper2)/low*100)
            leverage = (math.floor(100 / calc))
            return leverage
        if long:
            high = last_candle['close']
            lower2 = last_candle['lower2']
            calc = abs((lower2-high)/high*100)
            leverage = (math.floor(100 / calc))
            return leverage

        

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        open_trades = Trade.get_trades_proxy(is_open=True)
        partial_exits_list = [t for t in open_trades if t.nr_of_successful_exits > 0]
        partial_exits = len(partial_exits_list)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()


        if partial_exits>0 and self.config['max_open_trades']<=self.m_trades+partial_exits:
            self.config['max_open_trades'] += 1
        if self.config['max_open_trades']>self.m_trades+partial_exits:
            self.config['max_open_trades'] -= 1

        fibonacci_indices = [2, 3, 5, 8]

        for i, fib_index in enumerate(fibonacci_indices):
            if trade.is_short:
                threshold = last_candle['VWMA'] - last_candle['standard_deviation'] * fib_index
                if current_rate <= threshold and trade.nr_of_successful_exits == i and current_profit > 0:
                    return -(trade.stake_amount / fib_index)
            else:
                threshold = last_candle['VWMA'] + last_candle['standard_deviation'] * fib_index
                if current_rate >= threshold and trade.nr_of_successful_exits == i and current_profit > 0:
                    return -(trade.stake_amount / fib_index)

    def lev_check(self, dataframe):
        last_candle = dataframe.iloc[-1].squeeze()
        long = (last_candle['close'] > last_candle['upper1'])
        short = (last_candle['close'] <= last_candle['lower1'])
        if short:
            upper2 = last_candle['upper2']
            low = last_candle['close']
            calc = abs((low-upper2)/low*100)
            leverage = (math.floor(100 / calc))
            return leverage
        if long:
            high = last_candle['close']
            lower2 = last_candle['lower2']
            calc = abs((lower2-high)/high*100)
            leverage = (math.floor(100 / calc))
            return leverage

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        var_length = 950

        dataframe['hlc3'] = pta.hl2(high=dataframe['high'], low=dataframe['low'], close = dataframe['close'])

        dataframe['VWMA'] = pta.vwma(close = dataframe['hlc3'], volume = dataframe['volume'], length= var_length)

        dataframe['standard_deviation'] = stdev(close=dataframe['VWMA'], length=var_length)
        dataframe['deviation1'] = 1 * dataframe['standard_deviation']
        dataframe['deviation2'] = 2 * dataframe['standard_deviation']
 
        dataframe['lower1'] = dataframe['VWMA'] - dataframe['deviation1']
        dataframe['upper1'] = dataframe['VWMA'] + dataframe['deviation1']
        dataframe['lower2'] = dataframe['VWMA'] - dataframe['deviation2']
        dataframe['upper2'] = dataframe['VWMA'] + dataframe['deviation2']

        dataframe['Leverage'] = self.lev_check(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['upper1']) &
                (dataframe['close'].shift(1) <= dataframe['upper1'].shift(1)) &
                (dataframe['Leverage'] > 4)
            ),
            'enter_long'] = 0

        dataframe.loc[
            (
                (dataframe['close'] <= dataframe['lower1']) &
                (dataframe['close'].shift(1) > dataframe['lower1'].shift(1)) &
                (dataframe['Leverage'] > 4)
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] <= dataframe['lower1'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['close'] > dataframe['upper1'])
            ),
            'exit_short'] = 1

        return dataframe






