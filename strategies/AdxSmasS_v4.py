
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime, timedelta

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class AdxSmasS_v4(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AdxSmas.cs

    """

    INTERFACE_VERSION: int = 3

    can_short: bool = True



    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.15

    timeframe = '1h'

    use_custom_stoploss = True

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 6
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 12,
                "trade_limit": 20,
                "stop_duration_candles": 6,
                "max_allowed_drawdown": 0.05
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.03
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                            current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit < 0.03:
            return -1 # return a value bigger than the initial stoploss to keep using the initial stoploss

        desired_stoploss = current_profit / 2

        return max(min(desired_stoploss, 0.08), 0.025)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['long'] = ta.SMA(dataframe, timeperiod=6)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        dataframe.loc[(), ['enter_long', 'enter_tag']] = (0, 'no_long_enter')

        dataframe.loc[
            (
                    (dataframe['adx'] < 25) &
                    (qtpylib.crossed_above(dataframe['long'], dataframe['short']))

            ),
            'enter_short'] = 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'no_long_exit')

        dataframe.loc[
            (
                    (dataframe['adx'] > 25) &
                    (qtpylib.crossed_above(dataframe['short'], dataframe['long']))

            ),
            'exit_short'] = 1

        return dataframe