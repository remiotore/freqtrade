import numpy as np
import pandas as pd
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from freqtrade.persistence import Trade
from datetime import datetime


class newIchimokuStrategy(IStrategy):

    timeframe = '15m'

    minimal_roi = {
        "0": 1.0,
        "1000": 0
    }

    stoploss = -0.2

    process_only_new_candles = True
    use_exit_signal = True
    can_short = False
    use_custom_stoploss = True
    exit_profit_only = True


    def custom_stoploss(self, pair: str, Trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if Trade.stop_loss == Trade.initial_stop_loss:


            spanline_stoploss = stoploss_from_absolute(last_candle['stoploss_line'], current_rate)
            if np.isnan(spanline_stoploss):
                return None
            else:
                return spanline_stoploss


        if current_profit < 0.04:
            return -1 # return a value bigger than the initial stoploss to keep using the initial stoploss

        desired_stoploss = current_profit / 2

        return max(min(desired_stoploss, 0.05), 0.025)
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        risk = trade.open_rate - last_candle['stoploss_line']
        target = risk * 2

        
        dataframe.loc[
            (

                (current_profit >= target)
            ),
            'exit_long'] = 1

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:


        high = dataframe['high'].rolling(9).max()
        low = dataframe['low'].rolling(9).min()
        dataframe['conversion'] = (high + low) / 2

        high2 = dataframe['high'].rolling(26).max()
        low2 = dataframe['low'].rolling(26).min()
        dataframe['base'] = (high2 + low2) / 2

        dataframe['span_a'] = (dataframe['conversion'] + dataframe['base']) / 2

        high3 = dataframe['high'].rolling(52).max()
        low3 = dataframe['low'].rolling(52).min()
        dataframe['span_b'] = (high3 + low3) / 2

        dataframe['lagging'] = dataframe['close'].shift(-26)

        dataframe['sma'] = pta.sma(dataframe['close'], timeperiod = 14)

        dataframe['rsi'] = pta.rsi(dataframe['close'], length=14)

        dataframe['stoploss_line'] = dataframe[['span_a', 'span_b']].min(axis=1)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (

                (dataframe['volume'] > 0) &

                (dataframe['close'] >= dataframe['span_a']) &
                (dataframe['close'] >= dataframe['span_b']) &

                (dataframe['span_a'] > dataframe['span_b']) &

                (dataframe['conversion'] > dataframe['base']) & 

                (dataframe['lagging'] >= dataframe['span_a']) &              

                (qtpylib.crossed_above(dataframe['span_a'], dataframe['span_b']))
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:








        return dataframe
