
import talib.abstract as ta
import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy

class CustomStrategy(IStrategy):
    timeframe = '1h'
    stoploss = -0.10
    roi = {
        "0": 0.1,
    }
    trailing_stop = False

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'])

        dataframe['bb_lowerband'], dataframe['bb_middleband'], dataframe['bb_upperband'] = ta.BBANDS(dataframe['close'])

        dataframe['rsi'] = ta.RSI(dataframe['close'])

        dataframe['doji'] = ta.CDLDOJI(dataframe)

        dataframe['upper_wick'] = dataframe['high'] - dataframe['close']
        dataframe['lower_wick'] = dataframe['open'] - dataframe['low']
        dataframe['candle_length'] = dataframe['high'] - dataframe['low']

        sideways_market_pct = 0.08
        dataframe['sideways_market'] = False
        dataframe['sideways_start'] = np.nan

        for i in range(len(dataframe) - 12):
            max_price = dataframe['high'][i:i+12].max()
            min_price = dataframe['low'][i:i+12].min()
            price_amplitude = (max_price - min_price) / min_price

            if price_amplitude <= sideways_market_pct:

                bb_width = (dataframe.at[i + 11, 'bb_upperband'] - dataframe.at[i + 11, 'bb_lowerband']) / dataframe.at[i + 11, 'bb_middleband']

                rsi_neutral = 30 < dataframe.at[i + 11, 'rsi'] < 70

                if bb_width <= 0.1 and rsi_neutral:
                    dataframe.at[i + 11, 'sideways_market'] = True
                    dataframe.at[i + 11, 'sideways_start'] = i

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                dataframe['sideways_market'] &
                (dataframe['macd'] > 0) &
                (dataframe['upper_wick'] <= 0.4 * dataframe['candle_length']) &
                (dataframe['lower_wick'] <= 0.4 * dataframe['candle_length'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['sell'] = 0

        for i in range(len(dataframe) - 1):
            if dataframe.at[i, 'doji'] and dataframe.at[i - 1, 'doji']:
                dataframe.at[i+1, 'sell'] = 1

        for i in range(len(dataframe) - 1):
            if dataframe.at[i, 'sideways_market']:
                if dataframe.at[i+1, 'close'] < dataframe.at[i, 'sideways_start']:
                    dataframe.at[i+1, 'sell'] = 1

        return dataframe