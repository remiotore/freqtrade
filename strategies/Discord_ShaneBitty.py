
# --- Do not remove these libs ---
from freqtrade.strategy import ( IStrategy, IntParameter )
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta


class ShaneBitty(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "emaLongVal": 39,
        "emaShortVal": 16,
        "rsiBuy": 49,
        "rsiTimePeriodBuy": 13,
    }

    # Sell hyperspace params:
    sell_params = {
        "rsiSell": 83,
        "rsiTimePeriodSell": 14,
        "volVal": 2,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.218,
        "17": 0.078,
        "32": 0.017,
        "138": 0
    }

    # Stoploss:
    stoploss = -0.348

    # Optimal timeframe for the strategy
    timeframe = '5m'

    emaShortVal = IntParameter(5, 20, default=buy_params['emaShortVal'], space="buy")
    emaLongVal = IntParameter(21, 100, default=buy_params['emaLongVal'], space="buy")
    rsiBuy = IntParameter(0, 50, default=buy_params['rsiBuy'], space="buy")
    rsiTimePeriodBuy = IntParameter(8, 14, default=buy_params['rsiTimePeriodBuy'], space="buy")
    rsiTimePeriodSell = IntParameter(8, 14, default=sell_params['rsiTimePeriodSell'], space="sell")
    rsiSell = IntParameter(50, 100, default=sell_params['rsiSell'], space="sell")
    volVal = IntParameter(0, 10, default=sell_params['volVal'], space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['emaShort'] = ta.EMA(dataframe, timeperiod=self.emaShortVal.value)
        dataframe['emaLong'] = ta.EMA(dataframe, timeperiod=self.emaLongVal.value)
        dataframe['rsiBuy'] = ta.RSI(dataframe, timeperiod=self.rsiTimePeriodBuy.value)
        dataframe['rsiSell'] = ta.RSI(dataframe, timeperiod=self.rsiTimePeriodSell.value)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (qtpylib.crossed_below(dataframe['rsiBuy'], self.rsiBuy.value)) &
                    (qtpylib.crossed_above(dataframe['emaShort'], dataframe['emaLong']))
                )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsiSell'], self.rsiSell.value)) &
                (dataframe['volume'] > self.volVal.value) 
            ),
            'sell'] = 1
        return dataframe
