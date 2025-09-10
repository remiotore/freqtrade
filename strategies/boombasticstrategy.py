# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade, 
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
#from technical import qtpylib
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
import numpy as np
from freqtrade.strategy import stoploss_from_open
def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    df.reset_index(inplace=True)

    ha_open = [(df['open'][0] + df['close'][0]) / 2]
    [ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df) - 1)]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High'] = df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    df['HA_Low'] = df[['HA_Open', 'HA_Close', 'low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O'] = ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C'] = ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H'] = ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L'] = ta.EMA(df['HA_Low'], sml)

    return df

class BoomBasticStrategy(IStrategy):
    INTERFACE_VERSION = 3
    # NOTE: settings as of the 25th july 21
    # Buy hyperspace params:
    # NOTE: Good value (Win% ~70%), alot of trades
    # "buy_min_fan_magnitude_gain": 1.008 # NOTE: Very save value (Win% ~90%), only the biggest moves 1.008,
    buy_params = {
        'buy_trend_above_senkou_level': 1,
        'buy_trend_bullish_level': 6,
        'buy_fan_magnitude_shift_value': 3,
        'buy_min_fan_magnitude_gain': 1.002,
    }
    # Sell hyperspace params:
    # NOTE: was 15m but kept bailing out in dryrun
    sell_params = {'sell_trend_indicator': 'trend_close_2h'}
    # ROI table:
    minimal_roi = {'0': 0.059, '10': 0.037, '41': 0.012, '114': 0}
    # Stoploss:
    stoploss = -0.0575
    # stoploss = -0.275
    # Optimal timeframe for the strategy
    timeframe = '5m'
    startup_candle_count: int = 256
    process_only_new_candles = True
    trailing_stop = False
    # trailing_stop_positive = 0.002
    # trailing_stop_positive_offset = 0.025
    # trailing_only_offset_is_reached = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    can_short = True
    plot_config = {
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(255,76,46,0.2)',
            },
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'},
        },
        'subplots': {
            'fan_magnitude': {'fan_magnitude': {}},
            'fan_magnitude_gain': {'fan_magnitude_gain': {}},
        },
    }
    # def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     heikinashi = qtpylib.heikinashi(dataframe)
    #     dataframe["open"] = heikinashi["open"]
    #     dataframe["trend_open_5m"] = dataframe["open"]
    #     dataframe["trend_open_15m"] = ta.EMA(dataframe["open"], timeperiod=3)
    #     dataframe["trend_open_30m"] = ta.EMA(dataframe["open"], timeperiod=6)
    #     dataframe["trend_open_1h"] = ta.EMA(dataframe["open"], timeperiod=12)
    #     dataframe["trend_open_2h"] = ta.EMA(dataframe["open"], timeperiod=24)
    #     dataframe["trend_open_4h"] = ta.EMA(dataframe["open"], timeperiod=48)
    #     dataframe["trend_open_6h"] = ta.EMA(dataframe["open"], timeperiod=72)
    #     dataframe["trend_open_8h"] = ta.EMA(dataframe["open"], timeperiod=96)
    #     # dataframe['close'] = heikinashi['close']
    #     dataframe["high"] = heikinashi["high"]
    #     dataframe["low"] = heikinashi["low"]
    #     dataframe["trend_close_5m"] = dataframe["close"]
    #     dataframe["trend_close_15m"] = ta.EMA(dataframe["close"], timeperiod=3).astype("float16")
    #     dataframe["trend_close_30m"] = ta.EMA(dataframe["close"], timeperiod=6).astype("float16")
    #     dataframe["trend_close_1h"] = ta.EMA(dataframe["close"], timeperiod=12).astype("float16")
    #     dataframe["trend_close_2h"] = ta.EMA(dataframe["close"], timeperiod=24).astype("float16")
    #     dataframe["trend_close_4h"] = ta.EMA(dataframe["close"], timeperiod=48).astype("float16")
    #     dataframe["trend_close_6h"] = ta.EMA(dataframe["close"], timeperiod=72).astype("float16")
    #     dataframe["trend_close_8h"] = ta.EMA(dataframe["close"], timeperiod=96).astype("float16")
    #
    #     dataframe["fan_magnitude"] = (dataframe["trend_close_1h"] / dataframe["trend_close_8h"]).astype("float16")
    #     dataframe["fan_magnitude_gain"] = dataframe["fan_magnitude"] / dataframe["fan_magnitude"].shift(1).astype("float16")
    #
    #     # Ichimoku Cloud - Keep only necessary columns
    #     ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
    #     dataframe["senkou_a"] = ichimoku["senkou_span_a"].astype("float16")
    #     dataframe["senkou_b"] = ichimoku["senkou_span_b"].astype("float16")
    #
    #     dataframe["atr"] = ta.ATR(dataframe).astype("float16")
    #
    #     # Drop any intermediate columns no longer needed
    #     dataframe.drop(
    #         columns=["chikou_span", "tenkan_sen", "kijun_sen",
    #                  "leading_senkou_span_a", "leading_senkou_span_b",
    #                  "cloud_green", "cloud_red"],
    #         inplace=True,
    #         errors="ignore"
    #         )
    #
    #     return dataframe
    # def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     conditions = []
    #     # Trending market
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 1:
    #         conditions.append(dataframe["trend_close_5m"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_5m"] > dataframe["senkou_b"])
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 2:
    #         conditions.append(dataframe["trend_close_15m"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_15m"] > dataframe["senkou_b"])
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 3:
    #         conditions.append(dataframe["trend_close_30m"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_30m"] > dataframe["senkou_b"])
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 4:
    #         conditions.append(dataframe["trend_close_1h"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_1h"] > dataframe["senkou_b"])
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 5:
    #         conditions.append(dataframe["trend_close_2h"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_2h"] > dataframe["senkou_b"])
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 6:
    #         conditions.append(dataframe["trend_close_4h"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_4h"] > dataframe["senkou_b"])
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 7:
    #         conditions.append(dataframe["trend_close_6h"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_6h"] > dataframe["senkou_b"])
    #     if self.buy_params["buy_trend_above_senkou_level"] >= 8:
    #         conditions.append(dataframe["trend_close_8h"] > dataframe["senkou_a"])
    #         conditions.append(dataframe["trend_close_8h"] > dataframe["senkou_b"])
    #     # Trends bullish
    #     if self.buy_params["buy_trend_bullish_level"] >= 1:
    #         conditions.append(dataframe["trend_close_5m"] > dataframe["trend_open_5m"])
    #     if self.buy_params["buy_trend_bullish_level"] >= 2:
    #         conditions.append(dataframe["trend_close_15m"] > dataframe["trend_open_15m"])
    #     if self.buy_params["buy_trend_bullish_level"] >= 3:
    #         conditions.append(dataframe["trend_close_30m"] > dataframe["trend_open_30m"])
    #     if self.buy_params["buy_trend_bullish_level"] >= 4:
    #         conditions.append(dataframe["trend_close_1h"] > dataframe["trend_open_1h"])
    #     if self.buy_params["buy_trend_bullish_level"] >= 5:
    #         conditions.append(dataframe["trend_close_2h"] > dataframe["trend_open_2h"])
    #     if self.buy_params["buy_trend_bullish_level"] >= 6:
    #         conditions.append(dataframe["trend_close_4h"] > dataframe["trend_open_4h"])
    #     if self.buy_params["buy_trend_bullish_level"] >= 7:
    #         conditions.append(dataframe["trend_close_6h"] > dataframe["trend_open_6h"])
    #     if self.buy_params["buy_trend_bullish_level"] >= 8:
    #         conditions.append(dataframe["trend_close_8h"] > dataframe["trend_open_8h"])
    #     # Trends magnitude
    #     conditions.append(
    #         dataframe["fan_magnitude_gain"] >= self.buy_params["buy_min_fan_magnitude_gain"]
    #     )
    #     conditions.append(dataframe["fan_magnitude"] > 1)
    #     for x in range(self.buy_params["buy_fan_magnitude_shift_value"]):
    #         conditions.append(dataframe["fan_magnitude"].shift(x + 1) < dataframe["fan_magnitude"])
    #     if conditions:
    #         dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1
    #     return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Calculate Heikin-Ashi candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']
        dataframe['ha_close'] = heikinashi['close']
        # Populate trend_open columns with astype("float16")
        dataframe['trend_open_5m'] = dataframe['open'].astype('float16')
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3).astype('float16')
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6).astype('float16')
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12).astype('float16')
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24).astype('float16')
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48).astype('float16')
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72).astype('float16')
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96).astype('float16')
        # Populate trend_close columns with astype("float16")
        dataframe['trend_close_5m'] = dataframe['close'].astype('float16')
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3).astype('float16')
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6).astype('float16')
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12).astype('float16')
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24).astype('float16')
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48).astype('float16')
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72).astype('float16')
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96).astype('float16')
        # Fan magnitude and gain
        dataframe['fan_magnitude'] = (
            dataframe['trend_close_1h'] / dataframe['trend_close_8h']
        ).astype('float16')
        dataframe['fan_magnitude_gain'] = (
            dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)
        ).astype('float16')
        # Populate Ichimoku Cloud components with astype("float16")
        ichimoku = ftt.ichimoku(
            dataframe,
            conversion_line_period=20,
            base_line_periods=60,
            laggin_span=120,
            displacement=30,
        )
        dataframe['senkou_a'] = ichimoku['senkou_span_a'].astype('float16')
        dataframe['senkou_b'] = ichimoku['senkou_span_b'].astype('float16')
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a'].astype('float16')
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b'].astype('float16')
        # ATR
        dataframe['atr'] = ta.ATR(dataframe).astype('float16')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = []
        short_conditions = []
        # LONG Conditions (Trending market - Bullish)
        if self.buy_params['buy_trend_above_senkou_level'] >= 1:
            long_conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            long_conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 6:
            long_conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            long_conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])
        # LONG Bullish Trends
        if self.buy_params['buy_trend_bullish_level'] >= 1:
            long_conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])
        if self.buy_params['buy_trend_bullish_level'] >= 6:
            long_conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])
        # Fan magnitude for LONG
        long_conditions.append(
            dataframe['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain']
        )
        long_conditions.append(dataframe['fan_magnitude'] > 1)
        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            long_conditions.append(
                dataframe['fan_magnitude'].shift(x + 1) < dataframe['fan_magnitude']
            )
        # SHORT Conditions (Trending market - Bearish)
        if self.buy_params['buy_trend_above_senkou_level'] >= 1:
            short_conditions.append(dataframe['trend_close_5m'] < dataframe['senkou_a'])
            short_conditions.append(dataframe['trend_close_5m'] < dataframe['senkou_b'])
        if self.buy_params['buy_trend_above_senkou_level'] >= 6:
            short_conditions.append(dataframe['trend_close_4h'] < dataframe['senkou_a'])
            short_conditions.append(dataframe['trend_close_4h'] < dataframe['senkou_b'])
        # SHORT Bearish Trends
        if self.buy_params['buy_trend_bullish_level'] >= 1:
            short_conditions.append(dataframe['trend_close_5m'] < dataframe['trend_open_5m'])
        if self.buy_params['buy_trend_bullish_level'] >= 6:
            short_conditions.append(dataframe['trend_close_4h'] < dataframe['trend_open_4h'])
        # Fan magnitude for SHORT
        short_conditions.append(
            dataframe['fan_magnitude_gain'] <= 1 / self.buy_params['buy_min_fan_magnitude_gain']
        )
        short_conditions.append(dataframe['fan_magnitude'] < 1)
        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            short_conditions.append(
                dataframe['fan_magnitude'].shift(x + 1) > dataframe['fan_magnitude']
            )
        # Assign enter signals
        if long_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, long_conditions), ['enter_long', 'enter_tag']] = (1, 'Trending market - Bullish')
        if short_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, short_conditions), ['enter_short', 'enter_tag']] = (1, 'Trending market - Bearish')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_exit_conditions = []
        short_exit_conditions = []
        # LONG Exit (Bullish reversal weakening)
        long_exit_conditions.append(
            qtpylib.crossed_below(
                dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']]
            )
        )
        # SHORT Exit (Bearish reversal weakening)
        short_exit_conditions.append(
            qtpylib.crossed_above(
                dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']]
            )
        )
        # Assign exit signals
        if long_exit_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, long_exit_conditions), ['exit_long', 'enter_tag']] = (1, 'Bullish Reversal Weakening')
        if short_exit_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, short_exit_conditions), ['exit_short', 'enter_tag']] = (1, 'Bearish Reversal Weakening')
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # Clamp to the lowest of proposed_stake, max_stake, and your hard cap of 10_000
        return min(proposed_stake, max_stake, 2500)


# def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#     conditions = []
#     conditions.append(
#         qtpylib.crossed_below(
#             dataframe["trend_close_5m"], dataframe[self.sell_params["sell_trend_indicator"]]
#         )
#     )
#     if conditions:
#         dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1
#     return dataframe


class LeveragedIchiV2x(ichiV1):

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (entry_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 2.0


class LeveragedIchiV3x(ichiV1):

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 3.0


class LeveragedIchiV4x(ichiV1):

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 4.0


class LeveragedIchiV10x(ichiV1):

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # Clamp to the lowest of proposed_stake, max_stake, and your hard cap of 10_000
        return min(proposed_stake * 10, max_stake, 512)

    # def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     conditions = []
    #     conditions.append(
    #         qtpylib.crossed_below(
    #             dataframe["trend_close_5m"], dataframe[self.sell_params["sell_trend_indicator"]]
    #         )
    #     )
    #     if conditions:
    #         dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1
    #     return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 10.0


class LeveragedIchiV30x(ichiV1):

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 30.0


class LeveragedIchiVTenPercentMaxLeverage(ichiV1):

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return max_leverage * 0.1
