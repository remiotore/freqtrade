# --- Imports ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (DecimalParameter, IntParameter, RealParameter, CategoricalParameter,
                                stoploss_from_open, merge_informative_pair, stoploss_from_absolute)
from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.exc import InvalidRequestError
import logging # Optional: For better debugging

logger = logging.getLogger(__name__) # Optional: For better debugging

# --- Freqtrade Config Compatibility ---
# "trading_mode": "futures",
# "margin_mode": "isolated",
# "position_adjustment_enable": true,
# "stake_amount": "unlimited" # (Recommended)

class SRFuturesStrategy2(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '15m'
    can_short = True # Strategy supports shorting

    # --- Strategy Configuration ---
    stoploss = -0.50 # Initial large stoploss (will be overridden by custom_stoploss)
    use_custom_stoploss = True # Enable the custom_stoploss logic

    minimal_roi = {"0": 10.0} # Effectively disables ROI based exits

    trailing_stop = False # Disable Freqtrade's native trailing stop

    # --- Position Adjustment Configuration ---
    position_adjustment_enable = True # Enable pyramiding/DCA logic
    max_entry_position_adjustment = 2 # Allow up to 2 additional entries

    # --- Strategy Hyperparameters ---
    # Leverage setting
    leverage_value = 3.0 # Fixed leverage value
    # leverage = DecimalParameter(1.0, 10.0, default=3.0, decimals=1, space='buy', optimize=True) # Example if optimizing leverage

    # Support/Resistance detection window
    sr_window = IntParameter(20, 80, default=40, space='buy', optimize=True)

    # Position Adjustment (Pyramiding) parameters
    pa_profit_threshold = RealParameter(0.01, 0.04, default=0.02, space='buy', optimize=True) # Minimum profit to consider adding
    pa_add_factor = RealParameter(0.8, 1.2, default=1.0, space='buy', optimize=True) # Size multiplier for additional entries

    # Custom Stoploss parameters
    sl_atr_multiplier = RealParameter(1.0, 3.0, default=1.5, space='buy', optimize=True) # ATR multiplier for initial SL distance
    sl_breakeven_atr_multiplier = RealParameter(0.1, 0.5, default=0.2, space='buy', optimize=True) # ATR multiplier for breakeven SL buffer

    # Entry Confirmation parameters (Original - partly replaced for longs)
    entry_confirmation_atr_multiplier = RealParameter(0.3, 0.8, default=0.5, space='buy', optimize=True) # Original ATR confirmation distance
    entry_volume_confirmation_factor = RealParameter(1.1, 2.0, default=1.3, space='buy', optimize=True) # Original volume confirmation factor

    # Stochastic Confirmation (New parameter - can be optimized)
    stoch_oversold_threshold = IntParameter(20, 35, default=25, space='buy', optimize=True)
    # stoch_overbought_threshold = IntParameter(65, 80, default=75, space='sell', optimize=True) # Add if modifying short logic

    # --- Order Configuration ---
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False, # Let the bot manage stoploss logic
    }

    # --- Helper Function to Find S/R Levels ---
    def find_sr_levels(self, dataframe: DataFrame, window: int) -> Tuple[List[float], List[float]]:
        """
        Identifies potential support and resistance levels within a given window.
        Filters levels based on proximity to current price using ATR.
        """
        if dataframe.empty or len(dataframe) < 5 or 'atr' not in dataframe.columns:
            # logger.warning("DataFrame too short or missing ATR for S/R detection.") # Optional
            return [], []

        # Use .iloc[-window:] for safety if window > len(dataframe) though check above prevents it
        relevant_df = dataframe.iloc[-min(window, len(dataframe)):]
        lows = relevant_df['low']
        highs = relevant_df['high']

        # Find min low and max high in the window
        potential_s = lows.min()
        potential_r = highs.max()

        # Initialize lists (could add more sophisticated peak/valley detection here)
        supports = sorted([potential_s])
        resistances = sorted([potential_r])

        current_price = dataframe['close'].iloc[-1]
        last_atr = dataframe['atr'].iloc[-1]

        # Define minimum distance between levels and current price
        if pd.isna(last_atr) or last_atr <= 0:
             # Fallback if ATR is invalid: use a small percentage of the range
             min_distance = (potential_r - potential_s) * 0.05 if (potential_r - potential_s) > 0 else 1e-8
        else:
             min_distance = last_atr * 0.5 # Use 0.5 * ATR as minimum distance

        min_distance = max(min_distance, 1e-8) # Ensure min_distance is positive

        # Filter supports: must be below current price by at least min_distance
        supports = [s for s in supports if current_price - s > min_distance]
        # Filter resistances: must be above current price by at least min_distance
        resistances = [r for r in resistances if r - current_price > min_distance]

        return supports, resistances

    # --- Indicator Population ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculates necessary technical indicators.
        MODIFIED: Added Stochastic calculation.
        """
        # Calculate ATR
        try:
            dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'])
        except Exception as e:
            logger.error(f"Error calculating ATR for {metadata['pair']}: {e}")
            dataframe['atr'] = np.nan

        # Calculate Volume SMA
        try:
            # Ensure window size is valid
            vol_sma_window = max(1, self.sr_window.value // 2)
            dataframe['volume_sma'] = dataframe['volume'].rolling(vol_sma_window).mean()
        except Exception as e:
            logger.error(f"Error calculating Volume SMA for {metadata['pair']}: {e}")
            dataframe['volume_sma'] = np.nan

        # Calculate MACD
        try:
            macd_df = pta.macd(dataframe['close'])
            if macd_df is not None and not macd_df.empty:
                # Dynamically find column names as pta might change them
                mc = next((col for col in macd_df.columns if 'MACD_' in col and 'MACDs' not in col and 'MACDh' not in col), None)
                mcs = next((col for col in macd_df.columns if 'MACDs_' in col), None)
                mch = next((col for col in macd_df.columns if 'MACDh_' in col), None)

                if mc and mcs and mch:
                    dataframe['macd'] = macd_df[mc]
                    dataframe['macdsignal'] = macd_df[mcs]
                    dataframe['macdhist'] = macd_df[mch]
                else:
                    logger.warning(f"Could not find expected MACD columns in pta output for {metadata['pair']}.")
                    dataframe[['macd', 'macdsignal', 'macdhist']] = 0.0
            else:
                dataframe[['macd', 'macdsignal', 'macdhist']] = 0.0
        except Exception as e:
            logger.error(f"Error calculating MACD for {metadata['pair']}: {e}")
            dataframe[['macd', 'macdsignal', 'macdhist']] = 0.0

        # --- Calculate Stochastic (MODIFIED: Added this section) ---
        try:
            # Using default pta.stoch settings (k=14, d=3, smooth_k=3) - parameters can be optimized
            stoch = pta.stoch(dataframe['high'], dataframe['low'], dataframe['close'])
            if stoch is not None and not stoch.empty:
                # Dynamically find column names
                k_col = next((col for col in stoch.columns if 'STOCHk' in col), None)
                d_col = next((col for col in stoch.columns if 'STOCHd' in col), None)
                if k_col and d_col:
                    dataframe['stoch_k'] = stoch[k_col]
                    dataframe['stoch_d'] = stoch[d_col]
                else:
                    logger.warning(f"Could not find expected Stochastic columns in pta output for {metadata['pair']}.")
                    dataframe[['stoch_k', 'stoch_d']] = 50.0 # Neutral default
            else:
                 dataframe[['stoch_k', 'stoch_d']] = 50.0 # Neutral default
        except Exception as e:
            logger.error(f"Error calculating Stochastic for {metadata['pair']}: {e}")
            dataframe[['stoch_k', 'stoch_d']] = 50.0 # Neutral default on error

        # --- Initialize custom columns used for storing trade-specific data ---
        dataframe['support_level_trigger'] = np.nan
        dataframe['resistance_level_1_target'] = np.nan
        dataframe['resistance_level_2_target'] = np.nan
        dataframe['stop_loss_price_long'] = np.nan

        dataframe['resistance_level_trigger'] = np.nan
        dataframe['support_level_1_target'] = np.nan
        dataframe['support_level_2_target'] = np.nan
        dataframe['stop_loss_price_short'] = np.nan

        return dataframe

    # --- Entry Signal Logic ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generates buy/sell signals based on S/R bounce confirmation.
        MODIFIED: Long entry confirmation now uses Stochastic instead of MACD/Volume/ATR price distance.
        """
        df_len = len(dataframe)
        dataframe.loc[:, ['enter_long', 'enter_short', 'enter_tag']] = (0, 0, None)

        # --- Pre-checks ---
        required_cols = ['atr', 'stoch_k', 'stoch_d'] # MODIFIED: Added stoch checks
        if df_len < self.sr_window.value + 1 or not all(col in dataframe.columns for col in required_cols):
            # logger.info(f"DataFrame too short or missing required columns for entry logic ({metadata['pair']}).")
            return dataframe
        # Check for NaNs in critical recent data
        if dataframe[required_cols + ['close', 'low', 'high', 'open']].iloc[-2:].isnull().any().any():
            # logger.info(f"NaN values found in recent critical data for entry logic ({metadata['pair']}).")
            return dataframe

        current_candle = dataframe.iloc[-1]
        prev_candle = dataframe.iloc[-2]
        current_index = dataframe.index[-1] # Use index for safer loc assignment

        # --- Long Entry Logic (MODIFIED CONFIRMATION) ---
        # 1. Find previous support level (using data up to the *previous* candle)
        sr_df_long = dataframe.iloc[-(self.sr_window.value + 1):-1].copy()
        supports_long, _ = self.find_sr_levels(sr_df_long, self.sr_window.value)
        prev_support_long = None
        if supports_long:
            below_supports = [s for s in supports_long if s < prev_candle['close']]
            if below_supports: prev_support_long = max(below_supports) # Closest support below prev close

        # 2. Check if previous candle touched or slightly broke the support
        touched_support = False
        if prev_support_long is not None and prev_candle['low'] <= prev_support_long * 1.003: # Allow slight break
            touched_support = True

        # 3. Check for Confirmation on Current Candle if support was touched
        if touched_support:

            # --- Confirmation Conditions (Method 1: Green Candle + Stochastic Cross from Oversold) ---
            is_green_candle = current_candle['close'] > current_candle['open']

            stoch_confirmed = False
            # Use the hyperparameter for the threshold
            oversold_threshold = self.stoch_oversold_threshold.value
            # Check for Stochastic Golden Cross from Oversold Area
            if (prev_candle['stoch_k'] < prev_candle['stoch_d'] and # K was below D
                current_candle['stoch_k'] > current_candle['stoch_d'] and # K crossed above D NOW
                prev_candle['stoch_k'] < oversold_threshold): # K was in oversold zone previously
                    stoch_confirmed = True

            # --- Combine Confirmations ---
            if is_green_candle and stoch_confirmed:
                # 4. Calculate Entry, Stop Loss, Take Profit, RRR
                entry_price_long = current_candle['close'] * 1.0005 # Slightly above close for limit order

                # Initial SL based on support level and ATR
                sl_offset_long = current_candle['atr'] * self.sl_atr_multiplier.value
                sl_price_long = prev_support_long - sl_offset_long

                # Ensure SL is reasonably below entry price (e.g., min 0.5%)
                if sl_price_long >= entry_price_long * (1 - 0.005):
                    sl_price_long = entry_price_long * (1 - 0.005)

                # Find current resistance levels for potential Take Profit targets
                # Use data up to the *current* candle to find relevant resistances
                sr_df_current_long = dataframe.tail(self.sr_window.value).copy()
                _, current_resistances_long = self.find_sr_levels(sr_df_current_long, self.sr_window.value)
                res1_target_long = np.nan
                res2_target_long = np.nan
                if current_resistances_long:
                    # Find resistances above the calculated entry price
                    above_res = sorted([r for r in current_resistances_long if r > entry_price_long])
                    if len(above_res) >= 1: res1_target_long = above_res[0] # Closest resistance is TP1
                    if len(above_res) >= 2: res2_target_long = above_res[1] # Next resistance is TP2

                # Calculate Risk/Reward Ratio based on TP1
                rrr_long = 0.0
                if not pd.isna(res1_target_long) and res1_target_long > entry_price_long:
                    profit_long = res1_target_long - entry_price_long
                    loss_long = entry_price_long - sl_price_long
                    if loss_long > 1e-8: # Avoid division by zero or tiny loss values
                        rrr_long = profit_long / loss_long

                # 5. Final Entry Condition: RRR Check
                if rrr_long >= 2.0:
                    # Store calculated levels in dataframe for this candle (will be picked up by confirm_trade_entry)
                    dataframe.loc[current_index, 'support_level_trigger'] = prev_support_long
                    dataframe.loc[current_index, 'resistance_level_1_target'] = res1_target_long
                    if not pd.isna(res2_target_long): dataframe.loc[current_index, 'resistance_level_2_target'] = res2_target_long
                    dataframe.loc[current_index, 'stop_loss_price_long'] = sl_price_long
                    # Set the entry signal
                    dataframe.loc[current_index, 'enter_long'] = 1
                    dataframe.loc[current_index, 'enter_tag'] = f'SR_StochConfirmL_RRR_{rrr_long:.2f}' # Updated tag


        # --- Short Entry Logic (Original - Needs similar modification for consistency) ---
        # 1. Find previous resistance level
        sr_df_short = dataframe.iloc[-(self.sr_window.value + 1):-1].copy()
        _, resistances_short = self.find_sr_levels(sr_df_short, self.sr_window.value)
        prev_resistance_short = None
        if resistances_short:
            above_resistances = [r for r in resistances_short if r > prev_candle['close']]
            if above_resistances: prev_resistance_short = min(above_resistances) # Closest resistance above prev close

        # 2. Check if previous candle touched resistance
        touched_resistance = False
        if prev_resistance_short is not None and prev_candle['high'] >= prev_resistance_short * 0.997: # Allow slight break
            touched_resistance = True

        # 3. Check for Original Confirmation on Current Candle if resistance was touched
        if touched_resistance:
            # Original confirmation logic (ATR price dist, Volume, MACD) - Consider replacing with Stochastic Death Cross
            price_conf_short = (
                current_candle['close'] < current_candle['open'] and # Red candle
                current_candle['close'] < prev_resistance_short - (current_candle['atr'] * self.entry_confirmation_atr_multiplier.value)
            )
            # Volume check needs volume_sma from prev candle
            volume_conf_short = False
            if 'volume_sma' in dataframe.columns and not pd.isna(dataframe['volume_sma'].iloc[-2]):
                 volume_conf_short = (
                    current_candle['volume'] > prev_candle['volume'] * self.entry_volume_confirmation_factor.value or
                    current_candle['volume'] > dataframe['volume_sma'].iloc[-2] * 1.1 # Compare to prev SMA
                 )

            macd_ok_short = (current_candle['macdhist'] <= prev_candle['macdhist'] * 0.7) or \
                            (current_candle['macd'] < current_candle['macdsignal'])

            if price_conf_short and volume_conf_short and macd_ok_short:
                # 4. Calculate Entry, SL, TP, RRR for Short
                entry_price_short = current_candle['close'] * 0.9995 # Slightly below close

                sl_offset_short = current_candle['atr'] * self.sl_atr_multiplier.value
                sl_price_short = prev_resistance_short + sl_offset_short

                if sl_price_short <= entry_price_short * (1 + 0.005): # Ensure SL is above entry
                    sl_price_short = entry_price_short * (1 + 0.005)

                sr_df_current_short = dataframe.tail(self.sr_window.value).copy()
                current_supports_short, _ = self.find_sr_levels(sr_df_current_short, self.sr_window.value)
                sup1_target_short = np.nan; sup2_target_short = np.nan
                if current_supports_short:
                    below_sup = sorted([s for s in current_supports_short if s < entry_price_short], reverse=True) # Desc order
                    if len(below_sup) >= 1: sup1_target_short = below_sup[0] # Highest support below entry
                    if len(below_sup) >= 2: sup2_target_short = below_sup[1] # Next highest

                rrr_short = 0.0
                if not pd.isna(sup1_target_short) and sup1_target_short < entry_price_short:
                    profit_short = entry_price_short - sup1_target_short
                    loss_short = sl_price_short - entry_price_short
                    if loss_short > 1e-8: rrr_short = profit_short / loss_short

                # 5. Final Short Entry Condition
                if rrr_short >= 2.0:
                    dataframe.loc[current_index, 'resistance_level_trigger'] = prev_resistance_short
                    dataframe.loc[current_index, 'support_level_1_target'] = sup1_target_short
                    if not pd.isna(sup2_target_short): dataframe.loc[current_index, 'support_level_2_target'] = sup2_target_short
                    dataframe.loc[current_index, 'stop_loss_price_short'] = sl_price_short
                    dataframe.loc[current_index, 'enter_short'] = 1
                    # Keep original tag until short logic is updated
                    dataframe.loc[current_index, 'enter_tag'] = f'SR_ConfirmRejectS_RRR_{rrr_short:.2f}'

        return dataframe

    # --- Exit Signal Logic ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No custom exit signals generated here.
        Exits are handled by custom_stoploss and adjust_trade_position (for TPs).
        """
        dataframe.loc[:, ['exit_long', 'exit_short', 'exit_tag']] = (0, 0, None)
        return dataframe

    # --- Stake Amount Calculation ---
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
        Calculates the initial stake amount, reserving capital for potential position adjustments.
        Only effective when stake_amount is "unlimited".
        """
        # Calculate the total parts needed for initial entry + max adjustments
        # Initial entry = 1 part
        # Each adjustment = 1 * pa_add_factor part
        total_parts = 1.0
        # Loop for the number of *additional* entries allowed
        for _ in range(self.max_entry_position_adjustment):
            total_parts += 1.0 * self.pa_add_factor.value # Use the hyperparameter

        if total_parts <= 0: # Safety check
             total_parts = 1.0

        # Initial stake is 1 part out of the total calculated parts
        initial_stake_ratio = 1.0 / total_parts

        # Calculate the custom stake based on the ratio and max available capital
        custom_stake = max_stake * initial_stake_ratio

        # Ensure the stake is within the minimum and maximum limits
        if min_stake is not None:
            custom_stake = max(custom_stake, min_stake)
        # Already limited by max_stake in calculation, but double-check
        custom_stake = min(custom_stake, max_stake)

        # logger.debug(f"Custom stake for {pair}: Total parts={total_parts:.2f}, Ratio={initial_stake_ratio:.2f}, Stake={custom_stake:.4f}")
        return custom_stake

    # --- Leverage Setting ---
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Sets the leverage for the trade.
        """
        return min(self.leverage_value, max_leverage) # Use fixed leverage, capped by exchange max

    # --- Custom Stoploss Logic ---
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        """
        Manages the stoploss dynamically.
        - Uses the initial SL calculated at entry time.
        - Moves SL to breakeven + buffer after the first partial exit (TP1).
        """
        initial_stop_price = None
        trade_custom_data: Optional[Dict[str, Any]] = None
        # Determine the key for the initial stop loss based on trade direction
        sl_key = 'initial_stop_price_long' if not trade.is_short else 'initial_stop_price_short'

        # --- Safely access custom_data ---
        try:
            # Check if custom_data exists and get the initial stop price
            if trade.custom_data:
                trade_custom_data = trade.custom_data
                initial_stop_price = trade_custom_data.get(sl_key)
        except InvalidRequestError:
            # logger.warning(f"Could not access trade.custom_data for trade {trade.id}, likely due to DB session issue.")
            initial_stop_price = None
            trade_custom_data = None
        except Exception as e:
            logger.error(f"Unexpected error accessing trade.custom_data for trade {trade.id}: {e}")
            return None # Return None to keep existing SL if error occurs

        # --- Get latest ATR ---
        # Use the dataprovider to get the analyzed dataframe
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or 'atr' not in dataframe.columns or dataframe['atr'].iloc[-1:].isnull().any():
            # logger.warning(f"Could not get valid dataframe or ATR for {pair} in custom_stoploss.")
            # Fallback: If we can't get ATR, maybe keep existing SL? Or use a failsafe %?
            # Returning None keeps the *last known* stoploss value Freqtrade has.
            return None
        last_atr = dataframe['atr'].iloc[-1]

        # --- Retrieve initial_stop_price if not found in custom_data (e.g., after restart) ---
        if initial_stop_price is None:
             # Try to get it from the dataframe column where it was stored during entry signal generation
             sl_col = 'stop_loss_price_long' if not trade.is_short else 'stop_loss_price_short'
             # Find the last non-NaN value in that column
             valid_sl_series = dataframe[dataframe[sl_col].notna()][sl_col]
             if not valid_sl_series.empty:
                 initial_stop_price = valid_sl_series.iloc[-1]
                 # Optionally store it back to custom_data if it was missing
                 # if trade_custom_data is not None:
                 #     trade.custom_data[sl_key] = initial_stop_price


        # --- Stoploss Adjustment Logic ---
        # Check if the first take profit has been hit (indicated by successful exits)
        tp1_hit = trade.nr_of_successful_exits > 0

        if tp1_hit:
            # --- Move SL to Breakeven + Buffer ---
            try:
                # Estimate total fees (can be refined)
                fee_ratio = trade.fee_open + trade.fee_close
                # Calculate ATR buffer for breakeven
                atr_buffer = last_atr * self.sl_breakeven_atr_multiplier.value

                # Calculate breakeven price including fees and ATR buffer
                if trade.is_short:
                    # For shorts, breakeven is below open rate
                    breakeven_price = trade.open_rate * (1 - fee_ratio) - atr_buffer
                    # Ensure SL price is slightly above current rate if breakeven is too low
                    if breakeven_price <= current_rate: breakeven_price = current_rate * (1 + 0.001)
                else:
                    # For longs, breakeven is above open rate
                    breakeven_price = trade.open_rate * (1 + fee_ratio) + atr_buffer
                    # Ensure SL price is slightly below current rate if breakeven is too high
                    if breakeven_price >= current_rate: breakeven_price = current_rate * (1 - 0.001)

                # Convert the absolute breakeven price to a relative stoploss percentage
                sl_new = stoploss_from_absolute(breakeven_price, current_rate, is_short=trade.is_short)

                # Ensure the calculated stoploss is actually a loss (negative for long, positive for short)
                # Protect against calculation errors leading to immediate stop out
                if trade.is_short and sl_new <= 0: sl_new = 0.0001 # Tiny positive SL for short
                elif not trade.is_short and sl_new >= 0: sl_new = -0.0001 # Tiny negative SL for long

                # logger.debug(f"{pair} TP1 hit. Moving SL to breakeven: Rate={current_rate:.4f}, BE_Price={breakeven_price:.4f}, SL%={sl_new:.4f}")
                return sl_new
            except Exception as e:
                logger.error(f"Error calculating breakeven stoploss for {pair}: {e}")
                return None # Keep existing SL on error
        elif initial_stop_price is not None and not np.isnan(initial_stop_price):
            # --- Use Initial Stoploss ---
            # Check if the initial stop price is still valid (SL hasn't been crossed yet)
            valid_sl = (trade.is_short and current_rate < initial_stop_price) or \
                       (not trade.is_short and current_rate > initial_stop_price)

            if valid_sl:
                try:
                    # Convert the absolute initial stop price to a relative percentage
                    sl_new = stoploss_from_absolute(initial_stop_price, current_rate, is_short=trade.is_short)

                    # Ensure calculated stoploss is valid direction (negative for long, positive for short)
                    if trade.is_short and sl_new <= 0: sl_new = 0.0001
                    elif not trade.is_short and sl_new >= 0: sl_new = -0.0001

                    # logger.debug(f"{pair} Using initial SL: Rate={current_rate:.4f}, SL_Price={initial_stop_price:.4f}, SL%={sl_new:.4f}")
                    return sl_new
                except Exception as e:
                    logger.error(f"Error calculating stoploss from initial price for {pair}: {e}")
                    return None # Keep existing SL on error
            else:
                # If current rate has already passed the initial stop price, trigger stop immediately
                # logger.warning(f"{pair} Current rate {current_rate:.4f} already passed initial SL {initial_stop_price:.4f}. Triggering stop.")
                return 0.0001 if trade.is_short else -0.0001 # Trigger immediate small stop
        else:
            # logger.warning(f"Could not determine initial stop price for {pair} (Trade ID: {trade.id}). Keeping existing stoploss.")
            return None # Keep existing SL if initial SL cannot be determined

    # --- Position Adjustment (Take Profit & Pyramiding) ---
    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Handles Take Profit (partial exits) and Pyramiding (adding to position).
        - Takes 70% profit at TP1.
        - Takes remaining profit at TP2.
        - Adds to position if profit threshold, price condition, and MACD condition are met.
        """
        tp1_price = None; tp2_price = None; trade_custom_data: Optional[Dict[str, Any]] = None
        # Determine keys for TP levels based on trade direction
        tp1_key = 'resistance_level_1_target' if not trade.is_short else 'support_level_1_target'
        tp2_key = 'resistance_level_2_target' if not trade.is_short else 'support_level_2_target'

        # --- Safely access custom_data ---
        try:
            if trade.custom_data:
                trade_custom_data = trade.custom_data
                tp1_price = trade_custom_data.get(tp1_key)
                tp2_price = trade_custom_data.get(tp2_key)
        except InvalidRequestError:
            # logger.warning(f"Could not access trade.custom_data for adjust_trade_position (Trade ID: {trade.id}).")
            trade_custom_data = None
        except Exception as e:
            logger.error(f"Unexpected error accessing trade.custom_data for adjust_trade_position (Trade ID: {trade.id}): {e}")
            return None # Do nothing if error occurs

        # --- Get latest dataframe ---
        dataframe, last_updated = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty or 'macd' not in dataframe.columns: # MACD still used for pyramiding confirmation
             # logger.warning(f"Could not get valid dataframe or MACD for {trade.pair} in adjust_trade_position.")
             return None

        # --- Retrieve TP levels if not found in custom_data ---
        if trade_custom_data is None: # Only try dataframe if custom_data failed
            if tp1_price is None:
                valid_tp1_series = dataframe[dataframe[tp1_key].notna()][tp1_key]
                if not valid_tp1_series.empty: tp1_price = valid_tp1_series.iloc[-1]
            if tp2_price is None:
                valid_tp2_series = dataframe[dataframe[tp2_key].notna()][tp2_key]
                if not valid_tp2_series.empty: tp2_price = valid_tp2_series.iloc[-1]

        # --- Profit Taking Logic ---
        exit_stake_amount = None # Amount of stake to exit (base currency)

        # TP1: If no exits yet and price reaches TP1
        if trade.nr_of_successful_exits == 0 and tp1_price is not None and not np.isnan(tp1_price):
            if (not trade.is_short and current_rate >= tp1_price) or \
               (trade.is_short and current_rate <= tp1_price):
                # Take 70% profit
                exit_stake_amount = trade.stake_amount * 0.7
                # logger.info(f"{trade.pair} Reached TP1 ({tp1_price:.4f}). Exiting 70% ({exit_stake_amount:.4f} stake).")

        # TP2: If already exited once and price reaches TP2
        elif trade.nr_of_successful_exits > 0 and tp2_price is not None and not np.isnan(tp2_price):
             if (not trade.is_short and current_rate >= tp2_price) or \
                (trade.is_short and current_rate <= tp2_price):
                 # Exit remaining position (full current stake)
                 exit_stake_amount = trade.stake_amount
                 # logger.info(f"{trade.pair} Reached TP2 ({tp2_price:.4f}). Exiting remaining position ({exit_stake_amount:.4f} stake).")

        # If an exit signal is generated, return the amount
        if exit_stake_amount is not None:
            try:
                # Freqtrade expects negative stake for selling long, positive for buying back short
                return -exit_stake_amount if not trade.is_short else exit_stake_amount
            except Exception as e:
                logger.error(f"Error determining exit stake amount for {trade.pair}: {e}")
                return None


        # --- Pyramiding (Add to Position) Logic ---
        add_position = False
        # Check conditions: in profit, haven't reached max adjustments
        if current_profit > self.pa_profit_threshold.value and \
           trade.nr_of_successful_entries <= self.max_entry_position_adjustment: # <= because initial entry is 1

            # Price condition: Ensure price is not too close to TP1 (allow room for profit)
            price_condition = True
            if tp1_price is not None and not np.isnan(tp1_price):
                 # Leave some room (e.g., 1% buffer) before TP1
                 if not trade.is_short: price_condition = current_rate < tp1_price * 0.99
                 else: price_condition = current_rate > tp1_price * 1.01

            if price_condition:
                 # MACD condition: Ensure momentum is still favorable
                 last_candle = dataframe.iloc[-1]
                 macd_still_ok = False
                 if not pd.isna(last_candle['macd']) and not pd.isna(last_candle['macdsignal']):
                     if not trade.is_short:
                         # MACD histogram positive OR MACD line above signal line
                         macd_still_ok = (last_candle['macdhist'] > 0) or (last_candle['macd'] > last_candle['macdsignal'])
                     else:
                         # MACD histogram negative OR MACD line below signal line
                         macd_still_ok = (last_candle['macdhist'] < 0) or (last_candle['macd'] < last_candle['macdsignal'])

                 if macd_still_ok:
                     add_position = True
                     # logger.debug(f"{trade.pair} Pyramiding conditions met. Profit={current_profit:.2%}, Price OK, MACD OK.")

        # If pyramiding conditions are met, calculate stake to add
        if add_position:
            try:
                # Get initial stake amount from the first filled entry order
                filled_entries = trade.select_filled_orders(trade.entry_side)
                if not filled_entries:
                    # logger.warning(f"Cannot pyramid {trade.pair}: No filled entry orders found.")
                    return None
                # Base the add amount on the *first* entry's stake
                initial_stake = filled_entries[0].stake_amount
                # Calculate stake to add based on the factor
                stake_to_add = initial_stake * self.pa_add_factor.value

                # Apply limits (min_stake and max_stake apply to the *individual order*)
                if max_stake is not None and stake_to_add > max_stake:
                    stake_to_add = max_stake
                if min_stake is not None and stake_to_add < min_stake:
                    # logger.info(f"Pyramid amount {stake_to_add:.4f} for {trade.pair} is below min_stake {min_stake:.4f}. Skipping.")
                    return None # Don't add if below minimum
                if stake_to_add <= 0: # Safety check
                    return None

                # logger.info(f"{trade.pair} Adding to position. Initial stake={initial_stake:.4f}, Adding={stake_to_add:.4f}")
                # Return positive stake amount: Freqtrade handles buy for long, sell for short
                return stake_to_add
            except Exception as e:
                logger.error(f"Error calculating pyramiding stake for {trade.pair}: {e}")
                return None

        # No action if neither TP nor pyramiding conditions met
        return None

    # --- Trade Entry Confirmation ---
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Called just before placing the entry order.
        Used here to store the calculated S/R/TP/SL levels into the trade's custom_data.
        This makes the levels persistent for the duration of the trade.
        """
        trade: Optional[Trade] = kwargs.get('trade') # Get the trade object if available (might not be on first entry)

        # If it's the *first* entry for this potential trade, the trade object might not exist yet,
        # or custom_data might not be accessible immediately.
        # We rely on populate_entry_trend having stored the values in the *dataframe* for the current candle.
        # Freqtrade will associate this dataframe row's data with the trade when it's created.

        # Let's try to access the dataframe to get the values calculated in populate_entry_trend
        # This is safer than relying on the trade object being fully formed here.
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"Could not get dataframe in confirm_trade_entry for {pair}. Cannot store custom data.")
            return True # Allow entry, but data won't be stored

        # Get the data from the *last row* of the dataframe, corresponding to the entry signal candle
        last_candle_data = dataframe.iloc[-1]

        # Initialize an empty dict to store data
        custom_data_to_store = {}

        # Store relevant levels based on trade side
        if side == 'long':
            trigger = last_candle_data.get('support_level_trigger')
            tp1 = last_candle_data.get('resistance_level_1_target')
            tp2 = last_candle_data.get('resistance_level_2_target')
            sl = last_candle_data.get('stop_loss_price_long')

            if trigger is not None and not pd.isna(trigger): custom_data_to_store['support_level_trigger'] = trigger
            if tp1 is not None and not pd.isna(tp1): custom_data_to_store['resistance_level_1_target'] = tp1
            if tp2 is not None and not pd.isna(tp2): custom_data_to_store['resistance_level_2_target'] = tp2
            if sl is not None and not pd.isna(sl): custom_data_to_store['initial_stop_price_long'] = sl

        elif side == 'short':
            trigger = last_candle_data.get('resistance_level_trigger')
            tp1 = last_candle_data.get('support_level_1_target')
            tp2 = last_candle_data.get('support_level_2_target')
            sl = last_candle_data.get('stop_loss_price_short')

            if trigger is not None and not pd.isna(trigger): custom_data_to_store['resistance_level_trigger'] = trigger
            if tp1 is not None and not pd.isna(tp1): custom_data_to_store['support_level_1_target'] = tp1
            if tp2 is not None and not pd.isna(tp2): custom_data_to_store['support_level_2_target'] = tp2
            if sl is not None and not pd.isna(sl): custom_data_to_store['initial_stop_price_short'] = sl

        # If we successfully gathered data, try to attach it to the trade object
        # This might fail if the trade object isn't ready, but Freqtrade should handle it later.
        if custom_data_to_store and trade:
            try:
                if trade.custom_data:
                    trade.custom_data.update(custom_data_to_store)
                else:
                    trade.custom_data = custom_data_to_store
                # logger.debug(f"Stored custom data for {pair} entry: {custom_data_to_store}")
            except Exception as e:
                logger.error(f"Error setting custom_data in confirm_trade_entry for {pair}: {e}")

        # Always return True to allow the entry order placement
        # This function's primary role here is data persistence, not gating entry.
        return True

    # --- Trade Exit Confirmation ---
    # Optional: Can be used to add final checks before an exit order (TP or SL) is placed
    # def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
    #                        rate: float, time_in_force: str, exit_reason: str,
    #                        current_time: datetime, **kwargs) -> bool:
    #     # Example: Prevent stoploss exit if price recovers quickly (use with caution!)
    #     # if exit_reason == 'stop_loss':
    #     #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     #     if not dataframe.empty:
    #     #        last_close = dataframe['close'].iloc[-1]
    #     #        stop_loss_price = trade.stop_loss # The price level Freqtrade determined for SL
    #     #        if not trade.is_short and last_close > stop_loss_price * 1.001: # Price bounced slightly above SL price
    #     #            logger.info(f"Stoploss for {pair} potentially averted by quick bounce. Denying exit.")
    #     #            return False # Deny the stoploss exit for now
    #     #        elif trade.is_short and last_close < stop_loss_price * 0.999: # Price bounced slightly below SL price
    #     #            logger.info(f"Stoploss for {pair} potentially averted by quick bounce. Denying exit.")
    #     #            return False # Deny the stoploss exit for now
    #     return True # Default: always confirm exit