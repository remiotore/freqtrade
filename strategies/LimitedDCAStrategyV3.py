# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from functools import reduce
from freqtrade.persistence import Trade
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# --- Strategy Class ---
class LimitedDCAStrategyV3(IStrategy):
    """
    Strategy Overview (V3 with enhanced entry/exit, dynamic ATR DCA, scaled DCA, lockout):
    1. Initial Entry (Enhanced):
        - Bullish trend (EMA_fast > EMA_slow).
        - ADX indicates trend strength.
        - Oversold condition (RSI < buy_rsi_level).
        - Price near/below Lower Bollinger Band.
    2. DCA (Dollar Cost Averaging) using adjust_trade_position:
        - Limited number of DCA entries (max_total_entries).
        - Price drops by a dynamic amount (ATR based).
        - Bullish trend must still be valid (EMA check in adjust_trade_position).
        - RSI indicates a continued or renewed low level (RSI check in adjust_trade_position).
        - Subsequent DCA stakes can be scaled.
        - Lockout period after a DCA.
    3. Exits (Enhanced):
        - RSI > sell_rsi_level.
        - MACD bearish crossover.
        - Price near/above Upper Bollinger Band.
        - Stop-loss: Fixed percentage.
        - ROI table: Time-based profit take.
        - Trailing stop-loss.
    """

    INTERFACE_VERSION = 3
    minimal_roi = {"0": 0.15, "60": 0.05, "120": 0.02, "240": 0.01}
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    timeframe = '1h'

    # --- Hyperparameters ---
    # General
    max_total_entries = IntParameter(1, 5, default=3, space="buy", optimize=True)

    # Initial Entry
    buy_rsi_level = IntParameter(10, 40, default=30, space="buy", optimize=True)
    ema_fast_period = IntParameter(10, 50, default=20, space="buy", optimize=True)
    ema_slow_period = IntParameter(50, 200, default=100, space="buy", optimize=True)
    adx_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    adx_threshold_buy = IntParameter(15, 40, default=20, space="buy", optimize=True)

    # DCA
    dca_rsi_level = IntParameter(10, 45, default=35, space="buy", optimize=True) # For DCA conditions
    dca_atr_multiplier = DecimalParameter(0.5, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    atr_period_dca = IntParameter(5, 20, default=14, space="buy", optimize=True)
    dca_scale_factor = DecimalParameter(1.0, 2.0, default=1.0, decimals=2, space="buy", optimize=True)
    dca_lockout_candles = IntParameter(0, 10, default=3, space="buy", optimize=True)

    # Exit
    sell_rsi_level = IntParameter(60, 90, default=70, space="sell", optimize=True)


    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 200 # EMA100 is longest, MACD/ADX shorter

    order_types = {
        'entry': 'limit', 'exit': 'limit', 'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}

    def timeframe_to_minutes(self, timeframe: str) -> int:
        s = timeframe; # Shortened for brevity
        if s == '1m': return 1;  elif s == '3m': return 3; elif s == '5m': return 5
        elif s == '15m': return 15; elif s == '30m': return 30; elif s == '1h': return 60
        elif s == '2h': return 120; elif s == '4h': return 240; elif s == '6h': return 360
        elif s == '8h': return 480; elif s == '12h': return 720; elif s == '1d': return 1440
        return 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe[f'ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_period.value)
        dataframe[f'ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['atr_dca'] = ta.ATR(dataframe, timeperiod=self.atr_period_dca.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe[f'ema_fast'] > dataframe[f'ema_slow']) # Main Trend
        conditions.append(dataframe['rsi'] < self.buy_rsi_level.value)    # Oversold
        conditions.append(dataframe['adx'] > self.adx_threshold_buy.value) # Trend Strength
        conditions.append(dataframe['close'] <= dataframe['bb_lowerband'] * 1.015) # Dip confirmation (1.5% tolerance)
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
        else:
            dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(dataframe['rsi'] > self.sell_rsi_level.value) # Overbought
        conditions.append(qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])) # MACD Bearish Cross
        conditions.append(dataframe['close'] >= dataframe['bb_upperband'] * 0.985) # Overextension (1.5% tolerance)
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1
        else:
            dataframe['exit_long'] = 0
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              **kwargs) -> float:

        if current_profit > 0.005: return -1 # Don't DCA if already in small profit

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty: return -1
        last_candle = dataframe.iloc[-1].squeeze()

        # --- DCA Lockout Period ---
        if self.dca_lockout_candles.value > 0:
            last_dca_time_str = trade.custom_info.get('last_dca_time')
            if last_dca_time_str:
                try:
                    last_dca_time = datetime.fromisoformat(last_dca_time_str)
                    if current_time.tzinfo is not None and last_dca_time.tzinfo is None:
                        last_dca_time = last_dca_time.replace(tzinfo=timezone.utc)
                    elif current_time.tzinfo is None and last_dca_time.tzinfo is not None:
                         current_time = current_time.replace(tzinfo=timezone.utc)
                         last_dca_time = last_dca_time.astimezone(timezone.utc)

                    time_since_last_dca = current_time - last_dca_time
                    candle_duration_minutes = self.timeframe_to_minutes(self.timeframe)
                    if candle_duration_minutes > 0:
                        candles_since_last_dca = time_since_last_dca.total_seconds() / (candle_duration_minutes * 60)
                        if candles_since_last_dca < self.dca_lockout_candles.value:
                            return -1
                except Exception as e:
                    logger.error(f"Error processing last_dca_time for {trade.pair}: {e}")
                    return -1

        # --- Dynamic DCA Trigger based on ATR ---
        atr_value = last_candle.get('atr_dca', 0.0)
        if atr_value == 0.0 or trade.open_rate_avg == 0: return -1
        required_profit_drop_for_dca = -abs((atr_value * self.dca_atr_multiplier.value) / trade.open_rate_avg)
        if current_profit >= required_profit_drop_for_dca: return -1

        # --- Check Max Entries ---
        if trade.nr_of_successful_entries >= self.max_total_entries.value: return -1

        # --- DCA Conditions (Trend, RSI - these are simpler than initial entry for DCA) ---
        if not (last_candle[f'ema_fast'] > last_candle[f'ema_slow'] and \
                last_candle['rsi'] < self.dca_rsi_level.value):
            return -1

        # --- Calculate Stake Amount for Scaled DCA ---
        try:
            initial_stake_amount = trade.orders[0].stake_amount if trade.orders else self.walet.get_trade_stake_amount(trade.pair, None)
            if not initial_stake_amount or initial_stake_amount < min_stake:
                 initial_stake_amount = self.walet.get_trade_stake_amount(trade.pair, None)
                 if not initial_stake_amount or initial_stake_amount < min_stake:
                     initial_stake_amount = min_stake

            dca_order_number = trade.nr_of_successful_entries # Nth DCA (N=1 for 1st DCA)
            stake_amount_for_this_dca = initial_stake_amount * (self.dca_scale_factor.value ** dca_order_number)
            
            stake_amount = max(min_stake, stake_amount_for_this_dca)
            stake_amount = min(stake_amount, max_stake)

            if stake_amount < min_stake: return -1

            logger.info(
                f"DCA for {trade.pair}: Entry #{trade.nr_of_successful_entries + 1} (DCA #{dca_order_number}). "
                f"Profit: {current_profit:.2%}, Req.Drop: {required_profit_drop_for_dca:.2%}. "
                f"Proposing stake: {stake_amount:.2f}"
            )
            trade.custom_info['last_dca_time'] = current_time.isoformat()
            return stake_amount
        except Exception as e:
            logger.error(f"Error calculating stake for DCA on {trade.pair}: {e}")
            return -1