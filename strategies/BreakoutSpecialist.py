from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class BreakoutSpecialist(IStrategy):
    """
    BreakoutSpecialist
    author@: PiP Repunzel

    How to use it?

    > freqtrade download-data --timeframes 5m --timerange=20200101-20250117
    > freqtrade backtesting --export trades -s BreakoutSpecialist --timeframe 5m --timerange=20200101-20250117
    > freqtrade plot-dataframe -s BreakoutSpecialist --indicators1 resistance support --timeframe 5m --timerange=20200101-20250117

    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.1
    }

    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.1  # Dynamic stoploss with ATR will also be applied

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for breakout detection and trend validation.
        """
        atr_period = 10  # Adjusted for better balance of noise and accuracy in shorter timeframes
        atr = ta.ATR(dataframe, timeperiod=atr_period)
        dataframe['atr'] = atr
        dataframe['resistance'] = dataframe['high'].rolling(20).max()
        dataframe['support'] = dataframe['low'].rolling(20).min()
        dataframe['donchian_high'] = dataframe['high'].rolling(20).max()
        dataframe['donchian_low'] = dataframe['low'].rolling(20).min()
        adx_threshold_base = 20  # Base ADX threshold
        adx_dynamic_adjustment = dataframe['atr'] / dataframe['close'] * 100  # Adjust ADX based on recent volatility
        dataframe['adx_threshold'] = adx_threshold_base + adx_dynamic_adjustment
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['adx_signal'] = dataframe['adx'] > dataframe['adx_threshold']
        dataframe['supertrend'] = qtpylib.supertrend(dataframe, period=10, multiplier=3)
        dataframe['volume_mean'] = dataframe['volume'].rolling(10).mean()  # Adjusted rolling window for shorter timeframe
        dataframe['trailing_stop'] = dataframe['close'] - (atr * 3)  # Increased multiplier to reduce premature stops
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy when breakout conditions are met.
        """
        dataframe.loc[
            (
                # Breakout above resistance
                (dataframe['close'] > dataframe['resistance']) &
                # Volume confirms breakout
                (dataframe['volume'] > dataframe['volume_mean'] * 1.5) &
                # Ensure strong trend (ADX threshold)
                (dataframe['adx_signal'] == True)
            ),
            'buy'
        ] = 1

        # Optional re-entry on pullback
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['resistance'] * 0.98) &
                (dataframe['close'] < dataframe['resistance'] * 1.02) &
                (dataframe['volume'] > dataframe['volume_mean'])
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell when trend weakens or trailing stop is hit.
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['trailing_stop']) |  # Trailing stop triggered
                (dataframe['adx'] < dataframe['adx_threshold'] * 0.75) |  # Adjusted ADX threshold for weakening trend
                (dataframe['close'] > dataframe['entry_price'] * 1.2)  # Take profit
            ),
            'sell'
        ] = 1

        return dataframe
