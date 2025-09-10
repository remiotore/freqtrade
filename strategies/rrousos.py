import numpy as np  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair

class Trader_4_1726727454_3078(IStrategy):
    """
    Optimized Strategy using the best Hyperopt results.
    """

    # Buy hyperspace parameters
    buy_params = {
        "buy_cti_32": -0.8,
        "buy_rsi_32": 25,
        "buy_rsi_fast_32": 55,
        "buy_sma15_32": 0.979,
    }

    # Sell hyperspace parameters
    sell_params = {
        "sell_fastx": 56,
        "sell_loss_cci": 134,
        "sell_loss_cci_profit": 0.0,
    }

    # ROI table
    minimal_roi = {
        "0": 1
    }

    # Stoploss
    stoploss = -0.25

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Max open trades
    max_open_trades = 4

    # Timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators to the dataframe.
        """
        # Example indicators, modify as per your strategy
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sma15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = ta.CTI(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy conditions.
        """
        conditions = []

        conditions.append(dataframe['cti'] < self.buy_params['buy_cti_32'])
        conditions.append(dataframe['rsi'] < self.buy_params['buy_rsi_32'])
        conditions.append(dataframe['rsi'] > self.buy_params['buy_rsi_fast_32'])
        conditions.append(dataframe['sma15'] < self.buy_params['buy_sma15_32'])

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define sell conditions.
        """
        conditions = []

        conditions.append(dataframe['rsi'] > self.sell_params['sell_fastx'])
        conditions.append(dataframe['cti'] > self.sell_params['sell_loss_cci'])

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions), 'sell'] = 1
        return dataframe
