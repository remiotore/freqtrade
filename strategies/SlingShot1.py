
from freqtrade.strategy import IStrategy, merge_informative_pair
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Optional


class SlingShot1(IStrategy):
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        max_leverage = 20
        return max_leverage


    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    INTERFACE_VERSION: int = 3

    can_short: bool = True

    stoploss = -0.35

    timeframe = '1m'

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive_offset = 0.15  # Trigger positive stoploss once crosses above this percentage
    trailing_stop_positive = 0.08 # Sell asset if it dips down this much

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):

        return [(f"BTC/USDT:USDT", '1m')]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.dp:

            inf_tf = '1m'
            informative = self.dp.get_pair_dataframe(pair=f"BTC/USDT:USDT",
                                                     timeframe=inf_tf)

            informative['emaslow'] = ta.EMA(informative, timeperiod=62)
            informative['emafast'] = ta.EMA(informative, timeperiod=38)


            dataframe = merge_informative_pair(dataframe, informative,self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['emafast_1m'].shift(1) < dataframe['emaslow_1m'].shift(1))
                &(dataframe['emafast_1m'] >= dataframe['emaslow_1m'])    
            ),
            'enter_long'] = 1
            
        dataframe.loc[
            (

                (dataframe['emafast_1m'].shift(1) >= dataframe['emaslow_1m'].shift(1))
                &(dataframe['emafast_1m'] < dataframe['emaslow_1m'])    
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['emafast_1m'].shift(1) >= dataframe['emaslow_1m'].shift(1))
                &(dataframe['emafast_1m'] < dataframe['emaslow_1m'])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (

                (dataframe['emafast_1m'].shift(1) < dataframe['emaslow_1m'].shift(1))
                &(dataframe['emafast_1m'] >= dataframe['emaslow_1m'])    
            ),
            'exit_short'] = 1
            
        return dataframe