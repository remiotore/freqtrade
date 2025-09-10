
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime, date, timedelta
from technical.indicators import ichimoku, chaikin_money_flow
from freqtrade.exchange import timeframe_to_prev_date


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class ep3mas2(IStrategy):
    """

    author@: ??

    idea:
        this strategy is based on the link here:

        https://github.com/freqtrade/freqtrade-strategies/issues/95
    """



    minimal_roi = {


        "0": 100
    }

    stoploss = -0.99

    use_custom_stoploss = True






    timeframe = '5m'

    epwindow = IntParameter(2, 100, default=7, space='buy')


    eptarg = DecimalParameter(
        0, 4, decimals=4, default=2.25446, space='sell')
    epstop = DecimalParameter(
        0, 4, decimals=4, default=0.29497, space='sell')
    epcat1 = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',

                                          ], default='close', space='buy')
    epcat2 = CategoricalParameter(['open', 'high', 'low', 'close', 'volume',

                                          ], default='close', space='buy')
    epma1 = IntParameter(2, 100, default=25, space='buy')
    epma2 = IntParameter(2, 100, default=50, space='buy')
    epma3 = IntParameter(2, 100, default=100, space='buy')
    epma4 = IntParameter(2, 100, default=100, space='buy')
    epma5 = IntParameter(2, 100, default=100, space='buy')
    epma6 = IntParameter(2, 100, default=100, space='sell')








    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe['epm1'] = ta.EMA(dataframe,
                                         timeperiod=self.epma1.value)
        dataframe['epm2'] = ta.EMA(dataframe,
                                         timeperiod=self.epma2.value)
        dataframe['epm3'] = ta.EMA(dataframe,
                                         timeperiod=self.epma3.value)
        dataframe['epm4'] = ta.EMA(dataframe,
                                         timeperiod=self.epma4.value)
        dataframe['epm5'] = ta.EMA(dataframe,
                                         timeperiod=self.epma5.value)
        dataframe['epm6'] = ta.EMA(dataframe,
                                         timeperiod=self.epma6.value)










        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['epm1'].rolling(self.epwindow.value).min() > dataframe['epm2'].rolling(self.epwindow.value).max()) &
                (dataframe['epm2'].rolling(self.epwindow.value).min() > dataframe['epm3'].rolling(self.epwindow.value).max()) &
                (dataframe[self.epcat1.value].rolling(self.epwindow.value).min() > dataframe['epm4'].rolling(self.epwindow.value).max()) &



                qtpylib.crossed_above(dataframe[self.epcat2.value], dataframe['epm5']) &

                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['close'] > 1000000) &


                (dataframe['volume'] > 0)
            ),
            'sell'] = 1
        return dataframe
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)

        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        
        if not trade_candle.empty:





            set_stoploss = trade.open_rate - (trade.open_rate) * self.epstop.value
            if current_rate - set_stoploss <= 0.001:



                return -0.000001


            
        return 1

    def custom_sell(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)

        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        
        if not trade_candle.empty:

            epstoploss = trade_candle['epm6'].iloc[0]


            eptarget = trade.open_rate + (trade.open_rate) * self.eptarg.value



            if current_rate >= eptarget: #Let prices stabilize before setting



                return 'sell_ep3mas'

            
        return 0 