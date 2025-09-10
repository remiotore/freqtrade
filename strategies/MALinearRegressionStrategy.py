# pragma: no cover
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import numpy as np
import pandas as pd


class MALinearRegressionStrategy(IStrategy):
    """
    Strategy based on three moving averages and trend detection using linear regression.

    Timeframe: 1m
    Indicators:
      - 5-period SMA (5-minute average)
      - 15-period SMA (15-minute average)
      - 60-period SMA (60-minute average)
      - Linear regression slope of the last 50 close values

    Buy Signal:
      - Current price is below all three moving averages.
      - The regression slope is positive (indicating an upward trend).

    Sell Signal:
      - Current price is above all three moving averages.
      - The regression slope is negative (indicating a downward trend).

    Risk Management:
      - Minimal ROI (Take Profit): 2%
      - Stop-Loss: 5%
      - Trailing Stop-Loss: Enabled with a 1% positive offset and a 2% offset trigger.
    """

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.02  # 2% target profit
    }

    # Stoploss:
    stoploss = -0.05  # 5% stop loss

    # Trailing stop-loss settings:
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1% retracement allowed
    trailing_stop_positive_offset = 0.02  # Activate trailing stop after a 2% gain
    trailing_only_offset_is_reached = True

    timeframe = '1m'

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Adds several technical indicators to the given DataFrame.
        """
        # Calculate moving averages based on the close price.
        dataframe['ma_5'] = ta.SMA(dataframe['close'], timeperiod=5)
        dataframe['ma_15'] = ta.SMA(dataframe['close'], timeperiod=15)
        dataframe['ma_60'] = ta.SMA(dataframe['close'], timeperiod=60)

        # Calculate linear regression slope over the last 50 candles.
        # This will give an idea of the trend direction.
        dataframe['reg_slope'] = dataframe['close'].rolling(window=50).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
            raw=True
        )

        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generate buy signals:
         - When the close is below all three moving averages.
         - And the regression slope is positive (indicating an upward trend).
        """
        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['ma_5']) &
                    (dataframe['close'] < dataframe['ma_15']) &
                    (dataframe['close'] < dataframe['ma_60']) &
                    (dataframe['reg_slope'] > 0)
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generate sell signals:
         - When the close is above all three moving averages.
         - And the regression slope is negative (indicating a downward trend).
        """
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['ma_5']) &
                    (dataframe['close'] > dataframe['ma_15']) &
                    (dataframe['close'] > dataframe['ma_60']) &
                    (dataframe['reg_slope'] < 0)
            ),
            'sell'
        ] = 1
        return dataframe
