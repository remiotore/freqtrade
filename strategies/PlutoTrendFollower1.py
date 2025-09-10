import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class PlutoTrendFollower1(IStrategy):
    INTERFACE_VERSION: int = 3

    can_short = False
    minimal_roi = {"0": 0.1}
    stoploss = -0.30

    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.005
    trailing_only_offset_is_reached = True

    timeframe = "1h"


    order_types = {
        'entry': 'limit',
        'exit': 'market',
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1
            },




























        ]


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])

        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) &
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)),
            'enter_long'] = 1

        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) &
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) &
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)),
            'exit_long'] = 1

        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) &
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)),
            'exit_short'] = 1

        return dataframe


