



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from typing import Optional, Union
from datetime import datetime

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_open, informative)
from freqtrade.persistence import Trade


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class ElliotV8_IF3_futlo(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v0.1"

    can_short = True

    timeframe = '5m'

    use_custom_stoploss = False

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 400


    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """

    entry_params = {
        "leverage": 10,
        "base_nb_candles_buy": 14,
        "ewo_high": 2.327,
        "ewo_low": -19.988,
        "low_offset": 0.975,
        "rsi_buy": 69
    }

    exit_params = {
        "pHSL": -9.99,
        "pPF_1": 0.0026,
        "pSL_1": 0.0016,
        "pPF_2": 0.0132,
        "pSL_2": 0.0112,
        "base_nb_candles_sell": 24,
        "high_offset": 0.991,
        "high_offset_2": 0.997
    }

    minimal_roi = {
        "0": 0.215,
        "40": 0.032,
        "87": 0.016,
        "201": 0
    }


    stoploss = -0.32

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """

    """
    BEGIN hyperspace params
    """


    lev = IntParameter(0, 50, default=entry_params['leverage'], space='buy', optimize=False)

    base_nb_candles_buy = IntParameter(5, 80, default=entry_params['base_nb_candles_buy'], space='buy', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=entry_params['low_offset'], space='buy', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0,default=entry_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=entry_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=entry_params['rsi_buy'], space='buy', optimize=True)


    pHSL = DecimalParameter(-0.500, -0.020, default=exit_params['pHSL'], decimals=3, space='sell', optimize=False)

    pPF_1 = DecimalParameter(0.008, 0.020, default=exit_params['pPF_1'], decimals=3, space='sell', optimize=False)
    pSL_1 = DecimalParameter(0.008, 0.020, default=exit_params['pSL_1'], decimals=3, space='sell', optimize=False)

    pPF_2 = DecimalParameter(0.040, 0.100, default=exit_params['pPF_2'], decimals=3, space='sell', optimize=False)
    pSL_2 = DecimalParameter(0.020, 0.070, default=exit_params['pSL_2'], decimals=3, space='sell', optimize=False)

    base_nb_candles_sell = IntParameter(5, 80, default=exit_params['base_nb_candles_sell'], space='sell', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=exit_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=exit_params['high_offset_2'], space='sell', optimize=True)



    """
    END hyperspace params
    """


    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """

        return self.lev.value


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value * self.lev.value
        SL_1 = self.pSL_1.value * self.lev.value
        PF_2 = self.pPF_2.value * self.lev.value
        SL_2 = self.pSL_2.value * self.lev.value




        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL

        if (current_profit > PF_1):
            return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)
        else:

            return HSL


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

    @informative('1h')
    @informative('30m')
    @informative('15m')
    @informative('5m')
    def populate_indicators_inf(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        A decorator for populate_indicators_Nn(self, dataframe, metadata), allowing these functions to
        define informative indicators.

        Example usage:

            @informative('1h')
            def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
                return dataframe

        :param timeframe: Informative timeframe. Must always be equal or higher than strategy timeframe.
        :param asset: Informative asset, for example BTC, BTC/USDT, ETH/BTC. Do not specify to use
        current pair.
        :param fmt: Column format (str) or column formatter (callable(name, asset, timeframe)). When not
        specified, defaults to:
        * {base}_{quote}_{column}_{timeframe} if asset is specified. 
        * {column}_{timeframe} if asset is not specified.
        Format string supports these format variables:
        * {asset} - full name of the asset, for example 'BTC/USDT'.
        * {base} - base currency in lower case, for example 'eth'.
        * {BASE} - same as {base}, except in upper case.
        * {quote} - quote currency in lower case, for example 'usdt'.
        * {QUOTE} - same as {quote}, except in upper case.
        * {column} - name of dataframe column.
        * {timeframe} - timeframe of informative dataframe.
        :param ffill: ffill dataframe after merging informative pair.
        :param candle_type: '', mark, index, premiumIndex, or funding_rate
        """

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['EWO'] = EWO(dataframe, 50, 200)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0




        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['rsi_fast'] < 35) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            ), ['enter_long', 'enter_tag']] += (1, 'enter_long_1;')


        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            ), ['enter_long', 'enter_tag']] += (1, 'enter_long_2;')












        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        dataframe.loc[:, 'exit_tag'] = ''
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0




        dataframe.loc[
            (
                (dataframe['rsi'] > 50) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow']) &
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['volume'] > 0)
            ), ['exit_long', 'exit_tag']] += (1, 'exit_long_1;')


        dataframe.loc[
            (
                (dataframe['rsi_fast'] > dataframe['rsi_slow']) &
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            ), ['exit_long', 'exit_tag']] += (1, 'exit_long_2;')












        return dataframe


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif