
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone, time

class rad_testing(IStrategy):
    """
    idea:
        Relative Average Distance: Calculate distance of price from its simple moving average.
        Measure mean reversion by applying RSI formula to price differences.
        Go long when RAD < 25 short when > 75

        Base on article from Sofien Kaabar:
        https://kaabar-sofien.medium.com/the-relative-average-distance-a-new-mean-reversion-trading-indicator-c21f8f1986e3
    """

    timeframe = '30m'

    custom_info = {}

    minimal_roi = {
        "0": 0.39724,
        "90": 0.07604,
        "258": 0.05495,
        "791": 0.1
    }

    stoploss = -0.50

    trailing_stop = False
    process_only_new_candles = True
    use_custom_stoploss = True
    sell_profit_only = False

    startup_candle_count = 35

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        if self.custom_info and pair in self.custom_info and trade:


            if self.dp and self.dp.runmode.value in ('backtest', 'hyperopt'):
                relative_sl = self.custom_info[pair].loc[current_time]['atr_ema']

            else:

                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                       timeframe=self.timeframe)




                relative_sl = dataframe['atr_ema'].iat[-1]


            if current_profit >= 0.07:
                return ((current_rate - relative_sl *1) / current_rate) - 1
            if current_profit >= 0.03:
                return ((current_rate - relative_sl *2) / current_rate) - 1
            if current_profit >= 0.01:
                return ((current_rate - relative_sl *3) / current_rate) - 1

            return ((current_rate - relative_sl *5) / current_rate) - 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['runmode'].value in ('live', 'dry_run'):

            trades = Trade.get_trades([Trade.pair == metadata['pair'],
                                    Trade.open_date > datetime.utcnow() - timedelta(days=1),
                                    Trade.is_open.is_(False),
                        ]).all()

            sumprofit = sum(trade.close_profit for trade in trades)
            if sumprofit < 0:

                self.lock_pair(metadata['pair'], until=datetime.now(timezone.utc) + timedelta(hours=6))

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=8)
        dataframe['price_dif'] = dataframe['close'] - dataframe['sma']
        dataframe['rad'] = ta.RSI(dataframe['price_dif'], timeperiod=5)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=8)
        dataframe['atr_ema'] = ta.EMA(dataframe['atr'], timeperiod=5)


        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=13)
        dataframe['slow_dif'] = dataframe['close'] - dataframe['sma_slow']
        dataframe['slow_rad'] = ta.SMA(ta.RSI(dataframe['slow_dif'], timeperiod=5),timeperiod=8)


        self.custom_info[metadata['pair']] = dataframe[['date', 'atr','atr_ema']].copy().set_index('date')
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (









                (
                (qtpylib.crossed_below(dataframe['rad'], 23)) &
                (dataframe['rad'].shift(1) > 23) &
                (dataframe['rad'].shift(2) > 23) &
                (dataframe['sma'] > dataframe['close'])
                )
                                
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (

                (
                    (qtpylib.crossed_below(dataframe['rad'], dataframe['slow_rad'])) &
                    (dataframe['rad'].shift(1) > 60)
                )












             
            ),
            'sell'] = 1
        return dataframe

 