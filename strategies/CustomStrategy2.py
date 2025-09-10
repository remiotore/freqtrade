





import talib.abstract as ta
import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy

class CustomStrategy2(IStrategy):
    minimal_roi = { "0": -1 }

    timeframe = '1h'
    stoploss = -0.10
    roi = {
        "0": 0.1,
    }
    trailing_stop = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        sideways_market_pct = 0.08
        dataframe['sideways_market'] = False
        dataframe['sideways_low'] = np.nan

        for i in range(len(dataframe) - 12):
            sideways_start = i
            sideways_end = i + 11
            max_price = dataframe['high'][sideways_start:sideways_end + 1].max()
            min_price = dataframe['low'][sideways_start:sideways_end + 1].min()

            first_candle_range = (dataframe.at[sideways_start, 'high'] - dataframe.at[sideways_start, 'low']) / dataframe.at[sideways_start, 'low']
            last_candle_range = (dataframe.at[sideways_end, 'high'] - dataframe.at[sideways_end, 'low']) / dataframe.at[sideways_end, 'low']

            if (max_price - min_price) / min_price <= sideways_market_pct and first_candle_range <= sideways_market_pct and last_candle_range <= sideways_market_pct:
                dataframe.at[sideways_end, 'sideways_market'] = True
                dataframe.at[sideways_end, 'sideways_low'] = min_price

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                (dataframe['sideways_market'] == True) &
                (dataframe['close'] <= dataframe['sideways_low'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['sell'] = 0

        for i in range(len(dataframe) - 1):
            if dataframe.at[i, 'sideways_market']:
                if dataframe.at[i+1, 'close'] < dataframe.at[i, 'sideways_low']:
                    dataframe.at[i+1, 'sell'] = 1

        return dataframe
