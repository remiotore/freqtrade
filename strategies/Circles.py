
from functools import reduce
import re
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np

import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from pandas import DataFrame, Series, DatetimeIndex, merge
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from user_data.strategies.Custom_inidcators3 import RMI, TKE, laguerre, osc, vfi, vwmacd, mmar, VIDYA, madrid_sqz
from user_data.strategies.Custom_indicators2 import TA2
from user_data.strategies.custom_indicators import WaveTrend
import user_data.strategies.Custom_inidcators3 as CT3











class Circles(IStrategy):

    minimal_roi = {
      "0": 0.99



    }

    stoploss = -0.99

    timeframe = '15m'
    inf_timeframe = '15m'

    trailing_stop = False
    trailing_stop_positive = 0.337
    trailing_stop_positive_offset = 0.433
    trailing_only_offset_is_reached = True  # Disabled / not configured


    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    use_custom_stoploss = False

    startup_candle_count: int = 24

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    plot_config = {
        'main_plot': {
            'SAR': {'color': 'green'},




        },
        'subplots': {
            'MMAR' :{




                'KaufOSCI': {'color': 'yellow'},

            }
        }
    }


    """
    Informative Pair Definitions
    """

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        infs = {}
        for pair in pairs:
            inf_pair = self.getInformative(pair)

            if (inf_pair != ""):
                infs[inf_pair] = (inf_pair, self.inf_timeframe)

        informative_pairs = list(infs.values())


        return informative_pairs



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        curr_pair = metadata['pair']

        if (self.isBull(curr_pair)) or (self.isBear(curr_pair)):
            inf_pair = self.getInformative(curr_pair)


            inf_slow = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=self.inf_timeframe)
            inf_fast = self.dp.get_pair_dataframe(pair=inf_pair, timeframe=self.timeframe)

            dataframe['VIDYA'] = VIDYA(dataframe, length=18)
            dataframe['KaufOSCI'] = TA2.ER(dataframe, period=20)
            dataframe['SAR'] = TA2.SAR(dataframe, af=0.0061,amax=0.2) # This indicator is great since when (af) is less than 0.01 and amax as default
            dataframe['RMI'] = RMI(dataframe, length=25, mom=3)

            dataframe['rmi'] = RMI(dataframe, length=25, mom=3)
            dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
            dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)
            dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() <= 2, 1, 0)














        dataframe["leadMA"], dataframe["ma10_c"], dataframe["ma20_c"],dataframe["ma30_c"], dataframe["ma40_c"],dataframe["ma50_c"],dataframe["ma60_c"],dataframe["ma70_c"],dataframe["ma80_c"],dataframe["ma90_c"] = CT3.mmar(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        short_conditions = []
        long_conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        if self.isBull(metadata['pair']):

            long_conditions.append(dataframe['volume'] > 0)















            if long_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 0

        elif self.isBear(metadata['pair']):

            short_conditions.append(dataframe['volume'] > 0)















            if short_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_long'] = 0

        else:
            dataframe.loc[(dataframe['close'].notnull()), 'enter_long'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        short_conditions = []
        long_conditions = []
        dataframe.loc[:, 'exit_tag'] = ''

        if self.isBull(metadata['pair']):













            if long_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'exit_long'] = 0

        elif self.isBear(metadata['pair']):
















            if short_conditions:
                dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'exit_long'] = 0

        else:
            dataframe.loc[(dataframe['close'].notnull()), 'exit_long'] = 0

        return dataframe


    def isBull(self, pair):
        return re.search(".*(BULL|UP|[235]L)", pair)

    def isBear(self, pair):
        return re.search(".*(BEAR|DOWN|[235]S)", pair)

    def getInformative(self, pair) -> str:
        inf_pair = ""
        if self.isBull(pair):
            inf_pair = re.sub('(BULL|UP|[235]L)', '', pair)
        elif self.isBear(pair):
            inf_pair = re.sub('(BEAR|DOWN|[235]S)', '', pair)

        return inf_pair



















    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']







































        return 1
