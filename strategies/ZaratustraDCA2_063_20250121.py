import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class ZaratustraDCA2_063(IStrategy):
    """
        Personalized Trading Strategy with Risk Management and DCA

        This automated strategy combines risk management techniques with the use of multiple positions (DCA, Dollar Cost Averaging)
        to optimize market operations. It includes advanced protection tools such as cooldown periods and loss control, aiming to
        minimize risks and maximize profit opportunities.

        This strategy is inspired by ZaratustraV20, ImpulseV1, and other ideas I had in mind.

        NOTE: This strategy is under development. Some functions are currently disabled and others might still contain errors.
        Please proceed with caution and thoroughly test before using it in live trading.
    """


    ### Strategy parameters ###

    exit_profit_only = True # False option still in development
    use_custom_stoploss = False # custom_stoploss still in development
    trailing_stop = False
    ignore_roi_if_entry_signal = True
    can_short = True
    use_exit_signal = True
    stoploss = -0.10
    startup_candle_count: int = 100
    timeframe = '5m'

    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 1  # Maximum DCA multiplier

    # ROI table:
    minimal_roi = {}

    ### Hyperopt ###

    # protections
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 120, default=72, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    ### Protections ###
    @property
    def protections(self):
        """
            Defines the protections to apply during trading operations.
        """

        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot

    ### Dollar Cost Averaging (DCA) ###
    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
            Calculates the stake amount to use for a trade, adjusted dynamically based on the DCA multiplier.
            - The proposed stake is divided by the maximum DCA multiplier (`self.max_dca_multiplier`)
              to determine the adjusted stake.
            - If the adjusted stake is lower than the allowed minimum (`min_stake`), it is automatically increased
              to meet the minimum stake requirement.
        """

        # Calculates the adjusted stake amount based on the DCA multiplier.
        adjusted_stake = proposed_stake / self.max_dca_multiplier

        # Automatically adjusts to the minimum stake if it is too low.
        if adjusted_stake < min_stake:
            adjusted_stake = min_stake

        return adjusted_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
            Adjusts the trade position dynamically based on profit and NATR values.
            - Uses NATR to calculate an adjustment factor for scaling the position.
            - Reduces position by 50% in case of high profits (defined by a 10% threshold adjusted by NATR).
            - Potential increase in position (currently disabled in code) is based on certain profit thresholds for each DCA level.
            - Calculates the new stake amount for additional DCA entries, increasing by 25% for each filled entry.
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        natr = dataframe['natr'].iloc[-1]

        # Define adjustment factor based on NATR
        natr_adjustment = 1 + (natr * 0.01)

        if trade.entry_side == "buy":
            if current_profit > (0.10 * natr_adjustment) and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)   # Reduce position by 50%

            if current_profit > (-0.04 * natr_adjustment) and trade.nr_of_successful_entries == 1:
#                return trade.stake_amount * 0.25  # Increase position by 25%
                return None

            if current_profit > (-0.06 * natr_adjustment) and trade.nr_of_successful_entries == 2:
#                return trade.stake_amount * 0.25  # Increase position by 25%
                return None

            if current_profit > (-0.08 * natr_adjustment) and trade.nr_of_successful_entries == 3:
#                return trade.stake_amount * 0.25  # Increase position by 25%
                return None

        else:
            if current_profit > (0.10 * natr_adjustment) and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)   # Reduce position by 50%

            if current_profit > (-0.04 * natr_adjustment) and trade.nr_of_successful_entries == 1:
#                return trade.stake_amount * 0.25  # Increase position by 25%
                return None

            if current_profit > (-0.06 * natr_adjustment) and trade.nr_of_successful_entries == 2:
#                return trade.stake_amount * 0.25  # Increase position by 25%
                return None

            if current_profit > (-0.08 * natr_adjustment) and trade.nr_of_successful_entries == 3:
#                return trade.stake_amount * 0.25  ## Increase position by 25%
                return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        try:
            stake_amount = filled_entries[0].cost

            if count_of_entries > 1:
                stake_amount = stake_amount * (
                            1 + (count_of_entries - 1) * 0.25)  # Increase by 25% for each additional entry

            return stake_amount

        except Exception as exception:
            logger.error(f"Error adjusting DCA position for the pair {trade.pair}: {exception}")
            return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
            Implements a dynamic trailing stoploss based on NATR:
            - Dynamically adjusts stoploss between -4% and -8% using the normalized NATR factor.
            - Only applies stoploss with losses if the final DCA level is reached.
            - Respects the global maximum stoploss of -10%.
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Calculation of NATR
        natr_min = dataframe['natr'].quantile(0.05)
        natr_max = dataframe['natr'].quantile(0.95)
        natr = dataframe['natr'].iloc[-1]

        # Normalize NATR between [natr_min, natr_max]
        natr_factor = min(max((natr - natr_min) / (natr_max - natr_min), 0), 1)
#        logger.warning(f"natr_factor: {natr_factor} // Last value of natr: {natr} ")

        # Calculates the dynamic stoploss within a range of -4% to -8%,
        # using the normalized NATR factor to adjust the stoploss level.
        trailing_stop_value = -0.04 - (natr_factor * (-0.08 + 0.04))
#        logger.warning(f"trailing_stop_value: {trailing_stop_value} ")

        # If the last level of DCA has not been reached, only allow stoploss with profits.
        if trade.nr_of_successful_entries < self.max_entry_position_adjustment:
            if current_profit < 0:
                return -1

        # If we have profits and are not at the last level of DCA.
        # Respect the maximum limit of -10% and generate the new stoploss
        return max(trailing_stop_value, -0.10)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
            Calculates technical indicators used to define entry and exit signals.
        """
        dataframe['natr'] = ta.NATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['dx']  = ta.SMA(ta.DX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['adx'] = ta.SMA(ta.ADX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['pdi'] = ta.SMA(ta.PLUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['mdi'] = ta.SMA(ta.MINUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
            Defines the conditions for long/short entries based on
            technical indicators such as ADX, PDI, and MDI.
        """
        df.loc[
            (
                    (qtpylib.crossed_above(df['dx'], df['pdi'])) &
                    (df['adx'] > df['mdi']) &
                    (df['pdi'] > df['mdi'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'ZaratustraDCA Entry Long')

        df.loc[
            (
                    (qtpylib.crossed_above(df['dx'], df['mdi'])) &
                    (df['adx'] > df['pdi']) &
                    (df['mdi'] > df['pdi'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'ZaratustraDCA Entry Short')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
            Defines exit conditions for trades based on the ADX indicator:
            - Exit Long: Triggered when 'dx' crosses below 'adx' and ADX is strong (>25).
            - Exit Short: Triggered when 'dx' crosses below 'adx' and ADX is weak (≤25).
        """
        df.loc[
            (
                    (qtpylib.crossed_below(df['dx'], df['adx'])) &
                    (df['adx'] > 25)  # Strong ADX (above 25, to confirm trend)

            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'ZaratustraDCA Exit Long')

        df.loc[
            (
                    (qtpylib.crossed_below(df['dx'], df['adx'])) &
                    (df['adx'] <= 25)  # Weak ADX (below 25, lacks trend strength)
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'ZaratustraDCA Exit Short')

        return df

