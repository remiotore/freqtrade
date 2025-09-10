from typing import Optional

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter
from pandas import DataFrame, Series
from datetime import datetime


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class ClucHAnix_5m_0(IStrategy):
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """

    buy_params = {
        "bbdelta_close": 0.01889,
        "bbdelta_tail": 0.72235,
        "close_bblower": 0.0127,
        "closedelta_close": 0.00916,
    }

    sell_params = {

        "pHSL": -0.10,
        "pPF_1": 0.025,
        "pPF_2": 0.035,
        "pSL_1": 0.02,
        "pSL_2": 0.03,

    }

    minimal_roi = {
        "0": 0.038,
        "10": 0.028,
        "40": 0.015,
    }

    stoploss = -0.99  # use custom stoploss

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'emergencysell': 'limit',
        'forcebuy': "limit",
        'forcesell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    bbdelta_close = RealParameter(0.0005, 0.04, default=0.01965, space='buy', optimize=True)
    closedelta_close = RealParameter(0.0005, 0.04, default=0.00556, space='buy', optimize=True)
    bbdelta_tail = RealParameter(0.7, 1.5, default=0.95089, space='buy', optimize=True)
    close_bblower = RealParameter(0.0005, 0.04, default=0.00799, space='buy', optimize=True)

    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', load=True)

    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    pPF_2 = DecimalParameter(0.010, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.010, 0.070, default=0.040, decimals=3, space='sell', load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value




        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'buy_tag'] = 'ClucHA'

        dataframe.loc[
            (
            (dataframe['lower'].shift().gt(0)) &
            (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value)) &
            (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value)) &
            (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value)) &
            (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
            ) |
            (
                     (dataframe['ha_close'] < dataframe['ema_slow']) &
                     (dataframe['ha_close'] < self.close_bblower.value * dataframe['bb_lowerband'])
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['volume'] > 0),
            'sell'
        ] = 0

        return dataframe


class Cluc5mDCA(ClucHAnix_5m_0):
    position_adjustment_enable = True

    max_rebuy_orders = 1
    max_rebuy_multiplier = 2

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:

        if (self.config['position_adjustment_enable'] is True) and (self.config['stake_amount'] == 'unlimited'):
            return proposed_stake / self.max_rebuy_multiplier
        else:
            return proposed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):

        if (self.config['position_adjustment_enable'] is False) or (current_profit > -0.08):
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        if 0 < count_of_buys <= self.max_rebuy_orders:
            try:

                stake_amount = filled_buys[0].cost

                stake_amount = stake_amount
                return stake_amount
            except Exception as exception:
                return None

        return None
