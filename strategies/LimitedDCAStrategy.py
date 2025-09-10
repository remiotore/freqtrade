# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from functools import reduce
from freqtrade.persistence import Trade
import logging

logger = logging.getLogger(__name__)

# --- Strategy Class ---
class LimitedDCAStrategy(IStrategy):
    """
    Strategy Overview:
    1. Initial Entry:
        - Bullish trend (EMA_fast > EMA_slow).
        - Oversold condition (RSI < buy_rsi_level).
    2. DCA (Dollar Cost Averaging) using adjust_trade_position:
        - Limited number of DCA entries (max_total_entries).
        - Price drops by a certain percentage from the current average price.
        - Bullish trend must still be valid.
        - RSI indicates a continued or renewed low level.
    3. Exits:
        - Profit take: RSI > sell_rsi_level.
        - Stop-loss: Fixed percentage.
        - ROI table: Time-based profit take.
        - Trailing stop-loss.
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # ROI table:
    minimal_roi = {
        "0": 0.15,    # If not exited by signal, try for 15%
        "60": 0.05,   # After 1 hour, 5% is fine
        "120": 0.02,  # After 2 hours, 2%
        "240": 0.01   # After 4 hours, 1%
    }

    # Stoploss:
    stoploss = -0.10  # 10% stoploss from average entry price

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01  # Trail if 1% profit is reached
    trailing_stop_positive_offset = 0.02  # Offset for trailing stop (2% below high)
    trailing_only_offset_is_reached = True # Only enable trailing stop if offset is reached

    # Timeframe
    timeframe = '1h'

    # --- Hyperparameters ---
    # Define maximum TOTAL number of entries for a trade (initial + DCAs).
    # E.g., if 3, it means 1 initial entry + up to 2 DCA entries.
    max_total_entries = IntParameter(1, 5, default=3, space="buy", optimize=True)

    # Initial Entry parameters
    buy_rsi_level = IntParameter(10, 40, default=30, space="buy", optimize=True)
    ema_fast_period = IntParameter(10, 50, default=20, space="buy", optimize=True)
    ema_slow_period = IntParameter(50, 200, default=100, space="buy", optimize=True)

    # DCA parameters
    # RSI level for considering a DCA entry
    dca_rsi_level = IntParameter(10, 45, default=35, space="buy", optimize=True)
    # Percentage drop from current average price to trigger DCA consideration.
    # Must be negative, e.g., -0.03 for a 3% drop.
    dca_trigger_pct_drop = DecimalParameter(-0.08, -0.01, default=-0.03, decimals=3, space="buy", optimize=True)

    # Exit parameters
    sell_rsi_level = IntParameter(60, 90, default=70, space="sell", optimize=True)

    # --- Strategy Settings ---
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False # Process ROI even if entry signal is present

    startup_candle_count: int = 200 # Based on longest indicator (EMA100 here, 200 is safe)

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False # Recommended to be False for custom stoploss logic
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several TAlib indicators to the given DataFrame.
        """
        # EMAs
        dataframe[f'ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_period.value)
        dataframe[f'ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands (optional, can be used for more conditions)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # print(f"Populated indicators for {metadata['pair']} | Last candle: {dataframe.iloc[-1]['date']}") # For debugging
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the 'enter_long' column with 1 for bullish entries.
        """
        conditions = []

        # Condition 1: Bullish Trend (Fast EMA > Slow EMA)
        conditions.append(dataframe[f'ema_fast'] > dataframe[f'ema_slow'])

        # Condition 2: Initial Entry Signal - RSI oversold
        conditions.append(dataframe['rsi'] < self.buy_rsi_level.value)

        # Optional: Price near BB lower band for dip confirmation
        # conditions.append(dataframe['close'] < dataframe['bb_lowerband'])

        # Condition 3: Volume check (basic example)
        conditions.append(dataframe['volume'] > 0) # Ensure there's trading activity

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        else:
            dataframe['enter_long'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the 'exit_long' column with 1 for sell signals.
        """
        conditions = []

        # Condition 1: Exit Signal - RSI overbought
        conditions.append(dataframe['rsi'] > self.sell_rsi_level.value)

        # Optional: Price touches BB upper band
        # conditions.append(dataframe['close'] > dataframe['bb_upperband'])

        # Condition 2: Volume check (basic example)
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        else:
            dataframe['exit_long'] = 0

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              **kwargs) -> float:
        """
        Custom trade adjustment logic, enabling DCA.
        This method is called when `position_adjustment_enable` is True in config.

        Parameters:
        - trade: The current open trade object.
        - current_time: Current time.
        - current_rate: Current asset price.
        - current_profit: Current profit percentage of the trade (based on average price).
        - min_stake: Minimum allowed stake for this pair.
        - max_stake: Maximum allowed stake for this pair (from config `max_entry_size` if set, else wallet balance).
        Returns:
        - Positive float: Amount of quote currency to use for an additional buy order.
        - Negative float or Zero (-1): Indicates no adjustment is desired.
        """

        # If already in decent profit, don't DCA.
        if current_profit > 0.005: # e.g., > 0.5% profit
            return -1

        # We are looking to average down, so current_rate should be lower than trade.open_rate (initial entry).
        # More importantly, current_profit should be negative, indicating a loss.
        if current_profit >= self.dca_trigger_pct_drop.value:
            # current_profit must be MORE negative than dca_trigger_pct_drop (which is negative)
            # e.g., if dca_trigger_pct_drop is -0.03 (-3%), current_profit must be -0.031 or lower.
            return -1

        # --- Check Max Entries ---
        # trade.nr_of_successful_entries counts the initial entry as 1.
        # If max_total_entries is 3, we allow entries 1, 2, 3.
        if trade.nr_of_successful_entries >= self.max_total_entries.value:
            logger.info(f"Max entries ({self.max_total_entries.value}) reached for {trade.pair}. No DCA.")
            return -1

        # --- Get DataFrame for current pair and time ---
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"Unable to get analyzed dataframe for {trade.pair} for DCA.")
            return -1
        last_candle = dataframe.iloc[-1].squeeze() # Current (most recent) candle's data

        # --- DCA Conditions ---
        dca_conditions = []
        # Condition 1: Trend is still bullish
        dca_conditions.append(last_candle[f'ema_fast'] > last_candle[f'ema_slow'])
        # Condition 2: RSI is still low or re-entered oversold/low territory for DCA
        dca_conditions.append(last_candle['rsi'] < self.dca_rsi_level.value)

        if not all(dca_conditions):
            # logger.info(f"DCA conditions not met for {trade.pair}: TrendOK={dca_conditions[0]}, RSI_OK={dca_conditions[1]}")
            return -1

        # --- Calculate Stake Amount for DCA ---
        # Use the initial stake amount for DCA entries.
        # Freqtrade will ensure this respects min_stake and does not exceed max_stake (from max_entry_size).
        try:
            # Get the configured stake amount. This is typically the 'stake_amount' from config.
            stake_amount = self.wallet.get_trade_stake_amount(trade.pair, None)

            # If stake_amount from wallet is None (e.g. in testing) or too small, try to use initial trade stake.
            if not stake_amount or stake_amount < min_stake:
                if trade.stake_amount and trade.nr_of_successful_entries > 0:
                    # Use the average stake of previous entries if possible
                    stake_amount = trade.stake_amount / trade.nr_of_successful_entries
                else: # Fallback to min_stake if initial stake info is not reliable
                    stake_amount = min_stake

            # Ensure stake is at least min_stake and at most max_stake (which Freqtrade might cap via max_entry_size)
            stake_amount = max(min_stake, stake_amount)
            # The function should return the *additional* stake. Freqtrade handles max_entry_size overall cap.
            # We just need to make sure this single DCA order isn't > max_stake for a single order.
            stake_amount = min(stake_amount, max_stake) # max_stake here is the limit for *this single additional order*

            if stake_amount < min_stake:
                logger.warning(f"DCA for {trade.pair}: Calculated stake {stake_amount} is below min_stake {min_stake}. Skipping DCA.")
                return -1

            logger.info(
                f"DCA for {trade.pair}: "
                f"Entries: {trade.nr_of_successful_entries}/{self.max_total_entries.value}, "
                f"Current Profit: {current_profit:.2%}, "
                f"Current Rate: {current_rate}, Avg Price: {trade.open_rate_avg:.4f}, "
                f"Signal RSI: {last_candle['rsi']:.1f}. "
                f"Proposing to add stake: {stake_amount}"
            )
            return stake_amount

        except Exception as e:
            logger.error(f"Error calculating stake for DCA on {trade.pair}: {e}")
            return -1 # Do not adjust if stake calculation fails

    # Optional: If you want to control stake amount for initial entries differently.
    # For this DCA strategy, adjust_trade_position primarily controls additional stake.
    # def custom_stake_amount(self, pair: str, current_time, current_rate: float,
    #                         proposed_stake: float, min_stake: float, max_stake: float,
    #                         entry_tag: str, **kwargs) -> float:
    #     # If it's an initial entry (not a DCA adjustment)
    #     # You could, for example, use a smaller initial stake
    #     # and then rely on DCA to build up the position.
    #     # trade = Trade.get_trades_proxy(pair=pair, is_open=True)
    #     # if not trade: # No open trade, so this is an initial entry
    #     #     return proposed_stake * 0.5 # Example: initial stake is half of configured
    #     return proposed_stake