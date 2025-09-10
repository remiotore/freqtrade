


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class bbl3h3rsisharpe(IStrategy):
    """
    Strategy: optimized for Sharpe ratio loss function

        Buy hyperspace params:
        {   'mfi-enabled': True,
            'mfi-value': 35,
            'rsi-enabled': True,
            'rsi-value': 31,
            'trigger': 'bb_lower3'}
        Sell hyperspace params:
        {   'sell-mfi-enabled': True,
            'sell-mfi-value': 53,
            'sell-rsi-enabled': False,
            'sell-rsi-value': 60,
            'sell-trigger': 'sell-bb_high3'}
        ROI table:
        {0: 0.47671, 142: 0.12234, 267: 0.02454, 1222: 0}
        Stoploss: -0.29309
    """

    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.47671,
        "142": 0.12234,
        "267": 0.02454,
        "1222": 0
    }


    stoploss = -0.29309

    trailing_stop = False




    ticker_interval = '1h'

    process_only_new_candles = False


    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 20

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached
        from the exchange.
        These pair/interval combinations are non-tradeable, unless they are
        part of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self,
                            dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        :param dataframe: Raw data from the exchange and parsed by
                          parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



































        dataframe['rsi'] = ta.RSI(dataframe)
























        dataframe['mfi'] = ta.MFI(dataframe)





        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe),
                                            window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger['lower']
        dataframe['bb_upperband3'] = bollinger['upper']



































































































        """

        if self.dp:
            if self.dp.runmode in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self,
                           dataframe: DataFrame,
                           metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given
        dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband3']) &
                (dataframe['rsi'] <= 31) &
                (dataframe['mfi'] <= 35)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self,
                            dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the
        given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband3']) &
                (dataframe['mfi'] >= 53)
            ),
            'sell'] = 1
        return dataframe
