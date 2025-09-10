from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


def supertrend(df: DataFrame, atr_period: int, factor: float) -> DataFrame:
    """
    Calculates the Supertrend indicator.
    :param df: DataFrame with price data (OHLC)
    :param atr_period: ATR period
    :param factor: Multiplier for ATR
    :return: DataFrame with 'supertrend' and 'direction' columns
    """
    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)

    supertrend = [False] * len(df)
    direction = [0] * len(df)

    for i in range(1, len(df)):
        if df['close'][i] > upperband[i - 1]:
            supertrend[i] = lowerband[i]
            direction[i] = 1
        elif df['close'][i] < lowerband[i - 1]:
            supertrend[i] = upperband[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]

            if direction[i] == 1 and lowerband[i] > supertrend[i]:
                supertrend[i] = lowerband[i]
            if direction[i] == -1 and upperband[i] < supertrend[i]:
                supertrend[i] = upperband[i]

    df[f'supertrend_{atr_period}_{factor}'] = supertrend
    df[f'direction_{atr_period}_{factor}'] = direction
    return df


class WhaleSupertrend(IStrategy):
    # ROI and stoploss configuration
    minimal_roi = {"0": 0.10}
    stoploss = -0.10
    timeframe = '1h'

    # Supertrend parameters
    atr_period_1 = 10
    factor_1 = 3.0
    atr_period_2 = 10
    factor_2 = 4.0
    atr_period_3 = 10
    factor_3 = 6.0
    atr_period_4 = 10
    factor_4 = 9.0
    atr_period_5 = 10
    factor_5 = 13.0
    atr_period_6 = 10
    factor_6 = 18.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for the strategy.
        Adds multiple Supertrend levels to the DataFrame.
        """
        # Apply Supertrend indicators
        dataframe = supertrend(dataframe, self.atr_period_1, self.factor_1)
        dataframe = supertrend(dataframe, self.atr_period_2, self.factor_2)
        dataframe = supertrend(dataframe, self.atr_period_3, self.factor_3)
        dataframe = supertrend(dataframe, self.atr_period_4, self.factor_4)
        dataframe = supertrend(dataframe, self.atr_period_5, self.factor_5)
        dataframe = supertrend(dataframe, self.atr_period_6, self.factor_6)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate buy signals based on Supertrend majority decision.
        """
        # Calculate bullish count (direction > 0 means uptrend)
        bullish_count = (
            (dataframe['direction_10_3.0'] > 0).astype(int) +
            (dataframe['direction_10_4.0'] > 0).astype(int) +
            (dataframe['direction_10_6.0'] > 0).astype(int) +
            (dataframe['direction_10_9.0'] > 0).astype(int) +
            (dataframe['direction_10_13.0'] > 0).astype(int) +
            (dataframe['direction_10_18.0'] > 0).astype(int)
        )

        # Generate buy signal if 3 or more Supertrends are bullish
        dataframe['buy_signal'] = (bullish_count >= 3).astype(int)

        # FreqTrade buy signal column
        dataframe.loc[dataframe['buy_signal'] == 1, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate sell signals based on Supertrend majority decision.
        """
        # Calculate bearish count (direction < 0 means downtrend)
        bearish_count = (
            (dataframe['direction_10_3.0'] < 0).astype(int) +
            (dataframe['direction_10_4.0'] < 0).astype(int) +
            (dataframe['direction_10_6.0'] < 0).astype(int) +
            (dataframe['direction_10_9.0'] < 0).astype(int) +
            (dataframe['direction_10_13.0'] < 0).astype(int) +
            (dataframe['direction_10_18.0'] < 0).astype(int)
        )

        # Generate sell signal if 3 or more Supertrends are bearish
        dataframe['sell_signal'] = (bearish_count >= 3).astype(int)

        # FreqTrade sell signal column
        dataframe.loc[dataframe['sell_signal'] == 1, 'exit_long'] = 1
        return dataframe
