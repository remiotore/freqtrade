


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade

import logging
logger = logging.getLogger(__name__)

class fixed_riskreward_loss_424(IStrategy):
    """
    This strategy uses custom_stoploss() to enforce a fixed risk/reward ratio
    by first calculating a dynamic initial stoploss via ATR - last negative peak

    After that, we caculate that initial risk and multiply it with an risk_reward_ratio
    Once this is reached, stoploss is set to it and sell signal is enabled

    Also there is a break even ratio. Once this is reached, the stoploss is adjusted to minimize
    losses by setting it to the buy rate + fees.
    """

    custom_info = {
        'risk_reward_ratio': 3.5,
        'set_to_break_even_at_profit': 1,
    }
    use_custom_stoploss = True
    stoploss = -0.9

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        """
            custom_stoploss using a risk/reward ratio
        """
        result = break_even_sl = takeprofit_sl = -1
        custom_info_pair = self.custom_info.get(pair)
        if custom_info_pair is not None:



            open_date_mask = custom_info_pair.index.unique().get_loc(trade.open_date_utc, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            if(len(open_df) != 1):
                return -1 # won't update current stoploss

            initial_sl_abs = open_df['stoploss_rate']

            initial_sl = initial_sl_abs/current_rate-1


            risk_distance = trade.open_rate-initial_sl_abs
            reward_distance = risk_distance*self.custom_info['risk_reward_ratio']


            take_profit_price_abs = trade.open_rate+reward_distance

            take_profit_pct = take_profit_price_abs/trade.open_rate-1

            break_even_profit_distance = risk_distance*self.custom_info['set_to_break_even_at_profit']

            break_even_profit_pct = (break_even_profit_distance+current_rate)/current_rate-1

            result = initial_sl
            if(current_profit >= break_even_profit_pct):
                break_even_sl = (trade.open_rate*(1+trade.fee_open+trade.fee_close) / current_rate)-1
                result = break_even_sl

            if(current_profit >= take_profit_pct):
                takeprofit_sl = take_profit_price_abs/current_rate-1
                result = takeprofit_sl

        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['atr'] = ta.ATR(dataframe)
        dataframe['stoploss_rate'] = dataframe['close']-(dataframe['atr']*2)
        self.custom_info[metadata['pair']] = dataframe[['date', 'stoploss_rate']].copy().set_index('date')



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: buys when SAR is smaller then candle before
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[:, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: does nothing
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[:, 'sell'] = 0
        return dataframe
