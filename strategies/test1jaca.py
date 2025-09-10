from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class ScalpingStrategy(IStrategy):
    # Minimal ROI designed for scalping
    minimal_roi = {
        "0": 0.01  # 1% target ROI
    }

    # Stoploss configuration
    stoploss = -0.01  # 1% stoploss

    # Optimal timeframe for scalping
    timeframe = '5m'

    # Use a maximum of 10% of available balance per trade
    stake_amount = 0.1

    # Indicators configuration
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA indicators
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=26)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

        # Bollinger Bands
        bbands = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_upperband'] = bbands['upperband']
        dataframe['bb_middleband'] = bbands['middleband']
        dataframe['bb_lowerband'] = bbands['lowerband']

        # MACD
        macd = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        return dataframe

    # Buy signal
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lowerband']) &  # Price near lower Bollinger Band
            (dataframe['rsi'] < 30) &  # RSI indicates oversold
            (dataframe['ema_fast'] > dataframe['ema_slow']),  # Uptrend signal
            'buy'] = 1
        return dataframe

    # Sell signal
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_upperband']) &  # Price near upper Bollinger Band
            (dataframe['rsi'] > 70) &  # RSI indicates overbought
            (dataframe['ema_fast'] < dataframe['ema_slow']),  # Downtrend signal
            'sell'] = 1
        return dataframe
