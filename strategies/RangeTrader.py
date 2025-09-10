from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class RangeTrader(IStrategy):
    """
    RangeTrader
    author@: Pip Rumpelstiltskin

    How to use it?

    > freqtrade download-data --timeframes 1h --timerange=20200101-20250117
    > freqtrade backtesting --export trades -s RangeTrader --timeframe 1h --timerange=20200101-20250117
    > freqtrade plot-dataframe -s RangeTrader --indicators1 support resistance --timeframe 1h --timerange=20200101-20250117

    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # minimal_roi = {
    #     "40": 0.0,
    #     "30": 0.01,
    #     "20": 0.02,
    #     "0": 0.04
    # }

    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.15  # Keep losses tight, ain't got time for bleeding

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # trailing stoploss
    trailing_stop = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate the indicators for support and resistance. 
        We're marking the battlegrounds here, guv. Keep your eye on these lines!
        """
        lookback_period = 20  # How far back we’re looking for levels

        # Volatility-based adjustment using ATR
        # ATR gives us a sense of the average range, handy for dynamic levels
        atr = ta.ATR(dataframe, timeperiod=14)
        dataframe['support'] = dataframe['low'].rolling(lookback_period).min() - atr
        dataframe['resistance'] = dataframe['high'].rolling(lookback_period).max() + atr
        dataframe['mid_point'] = (dataframe['support'] + dataframe['resistance']) / 2

        # Detecting trend markets using ADX
        # If ADX is low, we assume a range-bound market
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['in_range_market'] = dataframe['adx'] < 25  # Only trade if ADX is low

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy when we're near the bottom of the range. Buy low, mate, sell high! Bob’s your uncle.
        """
        dataframe.loc[
            (
                # Close price near support (fading the dip)
                (dataframe['close'] <= dataframe['support'] * 1.01) &
                # Make sure there's some action in the market, no ghosts
                (dataframe['volume'] > 0) &
                # Ensure we're in a range market
                (dataframe['in_range_market'] == True)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell when we’re near the top, or get out if the party’s over. Don’t let greed nick your profits!
        """
        dataframe.loc[
            (
                # Close price near resistance (taking profit at the top)
                (dataframe['close'] >= dataframe['resistance'] * 0.99) |
                # Or if price buggers off past support—don’t hang about
                (dataframe['close'] < dataframe['support'] * 0.98) |
                # Ensure we're not in a trending market
                (dataframe['in_range_market'] == False)
            ),
            'sell'] = 1

        # Implementing partial exit at the mid-point
        # This ensures we lock in some profits if the market reaches halfway
        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['mid_point']) &
                (dataframe['volume'] > 0)
            ),
            'sell_partial'] = 0.5  # Sell half position near mid-point

        return dataframe
