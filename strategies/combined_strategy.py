



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from typing import Optional
from datetime import datetime
import talib

class combined_strategy(IStrategy):
    """
    """


    INTERFACE_VERSION = 3

    can_short: bool = True


    minimal_roi = {
    "10": 0.02,
    "7": 0.035,
    "5": 0.055,
    "3": 0.15,
    "0": 0.30,

    }


    stoploss = -0.99

    
    trailing_stop = False
    trailing_stop_positive = 0.15
    trailing_stop_positive_offset = 0.25
    trailing_only_offset_is_reached = False

    timeframe = '15m'

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    buy_ema1 = IntParameter(low=1, high=25, default=9, space='buy', optimize=True, load=True)
    buy_ema2 = IntParameter(low=1, high=50, default=21, space='buy', optimize=True, load=True)
    sell_ema1 = IntParameter(low=1, high=25, default=21, space='sell', optimize=True, load=True)
    sell_ema2 = IntParameter(low=1, high=50, default=9, space='sell', optimize=True, load=True)

    startup_candle_count: int = 100

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
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['uo'] = ta.ULTOSC(dataframe)

        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_percent"] = (
        (dataframe["close"] - dataframe["kc_lowerband"]) /
        (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        )
        dataframe["kc_width"] = (
        (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        )

        dataframe['uo'] = ta.ULTOSC(dataframe)

        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        dataframe['cci'] = ta.CCI(dataframe)

        vol = talib.MA(dataframe['volume'], timeperiod=14)
        dataframe['vol_ma'] = vol

        dataframe['rsi'] = ta.RSI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']



        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=10, stds=1.5)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
       (dataframe["close"] - dataframe["bb_lowerband"]) /
       (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
       (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)

        dataframe['sma9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=15)



        dataframe['CDL3LINESTRIKE_15M'] = ta.CDL3LINESTRIKE(dataframe)

        dataframe['CDL3LINESTRIKE_15M'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]

        dataframe['CDLENGULFING_15M'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]

        dataframe['CDLHARAMI_15M'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]

        dataframe['CDL3OUTSIDE_15M'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]

        dataframe['CDL3INSIDE_15M'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]



        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']


        """

        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

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

            (dataframe['ema9'] > dataframe['ema21']) &
            (dataframe['ema9'].shift(1) <= dataframe['ema21'].shift(1)) &  # Guard: previous candle ema9 <= ema21
            (dataframe['ha_open'] > dataframe['ha_close'])  # green bar
            ),
            'enter_long'] = 1

        dataframe.loc[
        (

            (dataframe['ema9'] < dataframe['ema21']) &
            (dataframe['ema9'].shift(1) >= dataframe['ema21'].shift(1)) &  # Guard: previous candle ema9 >= ema21
            (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
        ),
            'enter_short'] = 1

        return dataframe
        

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (

            (dataframe['ema9'] < dataframe['ema21']) &
            (dataframe['ema9'].shift(1) >= dataframe['ema21'].shift(1)) &  # Guard: previous candle ema9 >= ema21
            (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
            ),

            'exit_long'] = 1

        dataframe.loc[
            (

            (dataframe['ema9'] > dataframe['ema21']) &
            (dataframe['ema9'].shift(1) <= dataframe['ema21'].shift(1)) &  # Guard: previous candle ema9 <= ema21
            (dataframe['ha_open'] > dataframe['ha_close'])  # green bar

            ),
            'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 12

        
        

