# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from freqtrade.strategy.strategy_helper import stoploss_from_absolute
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from ta.trend import stc
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.pivots_points import pivots_points



class FuturesStrat8(IStrategy):
    custom_info = {}
    """
    This is a strategy template to get you started.
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
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Can this strategy go short?
    can_short = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.166,
        "64": 0.104,
        "203": 0.042,
        "531": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.078

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    custom_info = {
        'risk_reward_ratio': 2.0,
        'set_to_break_even_at_profit': 0,
    }
    use_custom_stoploss = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = 200

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, **kwargs) -> float:
        return self.wallets.get_total_stake_amount() / 10

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['stoploss_rate_long'] = dataframe['low'].rolling(5).min() - (dataframe['low'].rolling(window=5).min() * 0.001)
        dataframe['stoploss_rate_short'] = dataframe['high'].rolling(5).max() + (dataframe['high'].rolling(window=5).min() * 0.001)
        self.custom_info[metadata['pair']] = dataframe[
            ['date', 'stoploss_rate_long', 'stoploss_rate_short']].copy().set_index('date')

        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['STC'] = stc(dataframe['close'], window_slow=50, window_fast=26, cycle=12, smooth1=3, smooth2=3)
        dataframe['STC_BULL'] = np.where(dataframe['STC'] > dataframe['STC'].shift(1), 1, 0)
        dataframe['STC_BER'] = np.where(dataframe['STC'] < dataframe['STC'].shift(1), 1, 0)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['adx_20'] = 20
        return dataframe

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
        return 10.0

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        """
            custom_stoploss using a risk/reward ratio
        """
        result = -1
        custom_info_pair = self.custom_info.get(pair)
        if custom_info_pair is not None:
            # using current_time/open_date directly via custom_info_pair[trade.open_daten]
            # would only work in backtesting/hyperopt.
            # in live/dry-run, we have to search for nearest row before it
            open_date_mask = custom_info_pair.index.unique().get_indexer([trade.open_date_utc], method='ffill')[0]
            open_df = custom_info_pair.iloc[open_date_mask]

            # trade might be open too long for us to find opening candle
            if (len(open_df) != 2):
                return -1  # won't update current stoploss

            if trade.is_short:
                if open_df['stoploss_rate_short'] == None:
                    return 0.0001
            else:
                if open_df['stoploss_rate_long'] == None:
                    return 0.0001


            initial_sl_abs = open_df['stoploss_rate_long']
            delta_price_for_stop_loss = trade.open_rate - initial_sl_abs
            fixed_take_profit_price = trade.open_rate + (
                    delta_price_for_stop_loss * self.custom_info['risk_reward_ratio'])

            if trade.is_short:
                initial_sl_abs = open_df['stoploss_rate_short']
                delta_price_for_stop_loss = initial_sl_abs - trade.open_rate
                fixed_take_profit_price = trade.open_rate - (
                            delta_price_for_stop_loss * self.custom_info['risk_reward_ratio'])

            fixed_stop_loss = stoploss_from_absolute(initial_sl_abs, current_rate, is_short=trade.is_short)

            result = fixed_stop_loss

            if trade.is_short:
                if current_rate < fixed_take_profit_price:
                    result = 0.0001
                    custom_info_pair.iloc[open_date_mask]['stoploss_rate_short'] = None
            else:
                if current_rate > fixed_take_profit_price:
                    result = 0.0001
                    custom_info_pair.iloc[open_date_mask]['stoploss_rate_long'] = None


            # if trade.is_short:
            #     print(f"short trade on pair {trade.pair} on {trade.open_date_utc} op: {trade.open_rate}, csl: {result}, slp: {initial_sl_abs}, tpp:{fixed_take_profit_price} cp: {current_rate}")
            # else:
            #     print(
            #         f"long trade on pair {trade.pair} on {trade.open_date_utc} op: {trade.open_rate}, csl: {result}, slp: {initial_sl_abs}, tpp:{fixed_take_profit_price} cp: {current_rate}")


        return result

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        dataframe.loc[
            (
                # LONG
                    (dataframe['low'] > dataframe['ema200']) &
                    (dataframe['STC_BULL'] > 0) &
                    (dataframe['STC_BER'].shift(1) > 0) &
                    (dataframe['STC'].shift(1) < 1) &
                    (dataframe['adx'] > 20) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                # SHORT
                    (dataframe['high'] < dataframe['ema200']) &  # Check if candle is winning
                    (dataframe['STC_BER'] > 0) &
                    (dataframe['STC_BULL'].shift(1) > 0) &
                    (dataframe['STC'].shift(1) > 95) &
                    (dataframe['adx'] > 20) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
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
                # (dataframe["close"] > dataframe["open"]) & # Exit if price reverse
                (dataframe['volume'] == 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 0
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)

        dataframe.loc[
            (
                # (dataframe["close"] < dataframe["open"]) & # Exit if price reverse
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 0

        return dataframe
