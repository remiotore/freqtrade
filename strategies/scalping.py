from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class scalping(IStrategy):
    minimal_roi = {
        "0": 0.01  # Target a 1% ROI
    }
    stoploss = -0.02  # A tight stop loss of 2%
    timeframe = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema5'] > dataframe['ema20']) &  # EMA bullish crossover
                (dataframe['rsi'] < 30)  # Oversold RSI
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema5'] < dataframe['ema20']) &  # EMA bearish crossover
                (dataframe['rsi'] > 70)  # Overbought RSI
            ),
            'sell'] = 1

        return dataframe
