import datetime
import talib.abstract as ta
from functools import reduce
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter


class EWOistV1(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.10

    buy_params = {
        "ewo_buy_high": -5.9,
        "ewo_buy_low": 16.4,
    }

    sell_params = {
        "ewo_sell_high": -16.6,
        "ewo_sell_low": 7.9,
    }

    minimal_roi = {}

    stoploss = -0.99

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.03  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    max_open_trades = -1

    ewo_buy_low   = DecimalParameter(-25.0, 25.0, space='buy',  default=buy_params.get('ewo_buy_low', 0),   decimals=1)
    ewo_buy_high  = DecimalParameter(-25.0, 25.0, space='buy',  default=buy_params.get('ewo_buy_high', 0),  decimals=1)
    ewo_sell_low  = DecimalParameter(-25.0, 25.0, space='sell', default=buy_params.get('ewo_sell_low', 0),  decimals=1)
    ewo_sell_high = DecimalParameter(-25.0, 25.0, space='sell', default=buy_params.get('ewo_sell_high', 0), decimals=1)

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {},
            'subplots' : {
                'Miscenalea' : {
                    'EWO' : { }
                }
            },
        }

        return plot_config
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['EWO'] = (ta.SMA(dataframe, timeperiod=50) - ta.SMA(dataframe, timeperiod=200)) / dataframe['close'] * 100

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_conditions = []

        buy_conditions.append(
            (dataframe['EWO'] > self.ewo_buy_low.value) & 
            (dataframe['volume'] > 0)
        )
        if buy_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, buy_conditions), 'enter_long'] = 1

        sell_conditions = []
        sell_conditions.append(
            (dataframe['EWO'] < self.ewo_buy_low.value) & 
            (dataframe['volume'] > 0)
        )
        if sell_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, sell_conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = []
        exit_long_conditions.append(
            (dataframe['EWO'] < self.ewo_sell_high.value) & 
            (dataframe['volume'] > 0)
        )
        if exit_long_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_long_conditions), 'exit_long'] = 1

        exit_short_conditions = []
        exit_short_conditions.append(
            (dataframe['EWO'] > self.ewo_sell_low.value) & 
            (dataframe['volume'] > 0)
        )
        if exit_short_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_short_conditions), 'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0