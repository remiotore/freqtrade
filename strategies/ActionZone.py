



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.persistence import Trade


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta

class ActionZone(IStrategy):


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 100000
    }


    stoploss = -1.00
    use_custom_stoploss = True

    trailing_stop = False




    timeframe = '1d'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    min_price_period: int = 14

    max_loss_per_trade = 10 # USD

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'fastMA': {
                'color': 'red',
                'fill_to': 'slowMA',
                'fill_color': 'rgba(232, 232, 232,0.2)'
            }, 
            'slowMA': {
                'color': 'blue',
            },
        },
    }
    

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        stoploss_price = last_candle['lowest']

        if current_profit == 0 and current_time - timedelta(minutes=1) < trade.open_date_utc:

            return (stoploss_price / current_rate) - 1

        return 1 # return a value bigger than the initial stoploss to keep using the initial stoploss

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        stop_price = last_candle['lowest']
        volume_for_buy = self.max_loss_per_trade / (current_rate - stop_price)
        use_money = volume_for_buy * current_rate

        return use_money

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        lowest = ta.MIN(dataframe, timeperiod=self.min_price_period)
        dataframe['lowest'] = lowest

        fastEMA = ta.EMA(dataframe, timeperiod=12)
        slowEMA = ta.EMA(dataframe, timeperiod=26)
        dataframe['fastMA'] = fastEMA
        dataframe['slowMA'] = slowEMA


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['fastMA'] > dataframe['slowMA']) &  # Bull
                (dataframe['close'] > dataframe['fastMA'] ) & # Price Cross Up
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        dataframe.loc[
            (
                (dataframe['fastMA'] < dataframe['slowMA']) & # Bear
                (dataframe['close'] < dataframe['fastMA'] ) & # Price Cross Down
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    
    

