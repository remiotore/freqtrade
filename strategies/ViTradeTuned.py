



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.strategy import merge_informative_pair


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.consensus import Consensus


PAIR_INF_TIMEFRAME = '1h'

class ViTradeTuned(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = False

    timeframe = '5m'

    process_only_new_candles = True

    trailing_stop = True
    trailing_stop_positive = 0.01032
    trailing_stop_positive_offset = 0.03518
    trailing_only_offset_is_reached = True













    minimal_roi = {
        "0": 0.109,
        "34": 0.056,
        "87": 0.032,
        "143": 0
    }

    stoploss = -0.1

    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.049  # value loaded from strategy
    trailing_stop_positive_offset = 0.13  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    buy_signal_buy_score_consensus = IntParameter(low=1, high=100, default=21, space='buy', optimize=True, load=True)
    buy_signal_sell_score_consensus = IntParameter(low=1, high=100, default=41, space='buy', optimize=True, load=True)
    buy_signal_buy_score_delta_threshold = DecimalParameter(low=0.01, high=0.1, default=0.03, space='buy', optimize=True, load=True)

    buy_signal_bb_perc = DecimalParameter(low=0.1, high=1, default=0.83, space='buy', optimize=True, load=True)

    buy_signal_bb_std_dev = IntParameter(low=1, high=3, default=1, space='buy', optimize=True, load=True)
    buy_signal_bb_window = IntParameter(low=8, high=300, default=83, space='buy', optimize=True, load=True)



    buy_signal_rolling_window_higher_tf = IntParameter(low=1, high=50, default=21, space='buy', optimize=True, load=True)
    buy_signal_skip_spikes_higher_tf = DecimalParameter(low=0.01, high=0.25, default=0.2, space='buy', optimize=True, load=True)

    sell_signal_sell_score_consensus = IntParameter(low=1, high=100, default=18, space='sell', optimize=True, load=True)
    sell_signal_bb_perc = DecimalParameter(low=0.1, high=0.5, default=0.3, space='sell', optimize=True, load=True)

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True
    startup_candle_count = 100

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
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
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=PAIR_INF_TIMEFRAME)

        informative['rolling_max'] = informative['close'].rolling(window=21).max()
        informative['roc'] = informative['close'].pct_change()

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, PAIR_INF_TIMEFRAME, ffill=True)

        c = Consensus(dataframe)
        c.evaluate_hull()
        c.evaluate_adx()
        c.evaluate_macd()
        c.evaluate_cci()
        c.evaluate_osc()
        c.evaluate_vwma()

        score = c.score()

        dataframe['score_sell'] = score['sell']
        dataframe['score_buy'] = score['buy']
        dataframe['buy_agreement'] = score['buy_agreement']
        dataframe['buy_disagreement'] = score['buy_disagreement']
        dataframe['sell_agreement'] = score['sell_agreement']
        dataframe['sell_disagreement'] = score['sell_disagreement']

        mid, lower = bollinger_bands(dataframe['low'], window_size=self.buy_signal_bb_window.value,
                                     num_of_std=self.buy_signal_bb_std_dev.value)
        dataframe['mid'] = np.nan_to_num(mid)
        dataframe['lower'] = np.nan_to_num(lower)
        dataframe['bbdelta'] = (dataframe['mid'] - dataframe['lower']).abs()
        dataframe['pricedelta'] = (dataframe['open'] - dataframe['close']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=self.buy_signal_bb_std_dev.value)
        dataframe['bb_low'] = bollinger['lower']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_perc'] = (dataframe['close'] - dataframe['bb_low']) / (dataframe['bb_upper'] - dataframe['bb_low'])

        dataframe['roc'] = dataframe['close'].pct_change()
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        dataframe.loc[
            (

                    (dataframe["close_"+PAIR_INF_TIMEFRAME].rolling(self.buy_signal_rolling_window_higher_tf.value).max().ge(dataframe['close']))
                    &

                    (dataframe['roc_'+PAIR_INF_TIMEFRAME] < self.buy_signal_skip_spikes_higher_tf.value)
                    &
                        (
                            dataframe['score_sell'].ge(self.buy_signal_sell_score_consensus.value) |
                            dataframe['score_buy'].ge(self.buy_signal_buy_score_consensus.value)
                        )
                    &
                    (dataframe['bb_perc'].le(self.buy_signal_bb_perc.value))
                    &
                    (dataframe['closedelta'].gt(dataframe['close'] * self.buy_signal_buy_score_delta_threshold.value))




                    &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                    dataframe["close_" + PAIR_INF_TIMEFRAME].rolling(
                        self.buy_signal_rolling_window_higher_tf.value).min().le(dataframe['close']) &
                    (dataframe['roc_' + PAIR_INF_TIMEFRAME].gt(-self.buy_signal_skip_spikes_higher_tf.value)) &
                    dataframe['score_buy'].le(self.buy_signal_sell_score_consensus.value) &
                    dataframe['bb_perc'].ge(1 - self.buy_signal_bb_perc.value) &
                    dataframe['closedelta'].lt(dataframe['close'] * self.buy_signal_buy_score_delta_threshold.value) &
                    dataframe['volume'] > 0
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['bb_perc'] > self.sell_signal_bb_perc.value) &
                (dataframe['score_sell'].ge(self.sell_signal_sell_score_consensus.value))
            ),

            'exit_long'] = 1

        dataframe.loc[
            (
                    dataframe['bb_perc'].lt(1 - self.sell_signal_bb_perc.value) &
                    dataframe['score_buy'].ge(self.sell_signal_sell_score_consensus.value)
            ),
            'exit_short'] = 1

        return dataframe


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)

    return rolling_mean, lower_band
