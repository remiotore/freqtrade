from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
import numpy as np
from datetime import datetime
import logging

class ScalpingMomentsStrategyV6(IStrategy):
    INTERFACE_VERSION = 3

    # Minimal ROI (Return on Investment) for exits
    minimal_roi = {"0": 0.015}  # 1.5% take-profit

    # Placeholder for stop loss (replaced by ATR-based stop in custom_stoploss)
    stoploss = -0.2

    # Trailing stop settings
    trailing_stop = True
    trailing_stop_positive = 0.001  # Trail at 0.1% profit
    trailing_stop_positive_offset = 0.0015  # Start trailing at 0.15% profit
    trailing_only_offset_is_reached = True
    # use_custom_stoploss = True

    # Timeframe for scalping (5-minute candles)
    timeframe = '5m'
    can_short = True

    # Risk management for futures
    position_adjustment_enable = False
    max_open_trades = 6  # Reduced from 10 to limit exposure
    trading_mode = "futures"
    margin_mode = "isolated"
    stake_amount = 'unlimited'  # Use dynamic stake sizing

    plot_config = {
        'main_plot': {
            # Indicators plotted on the main price chart
            'ema_20': {'color': 'orange'}, # EMA 20
            'ema_50': {'color': 'purple'}, # EMA 50
            'ghost_candle': {'color':'white'}
            # 'bb_upper': {'color': 'blue'}, # Bollinger Band Upper
            # 'bb_lower': {'color': 'blue'}, # Bollinger Band Lower
            # Optionally fill between BB
            # 'bb_middle': {'color': 'orange'}, # Bollinger Band Middle
        },
        'subplots': {
            # Subplots below the main chart
            "RSI": {
                'rsi': {'color': 'blue'},
                # Add dynamic thresholds if you want them plotted
                'rsi_upper': {'color': 'red', 'ls': '--'}, # Dynamic upper threshold
                'rsi_lower': {'color': 'green', 'ls': '--'} # Dynamic lower threshold
                # Or fixed lines
                # 70: {'color': 'red', 'ls': '--'},
                # 30: {'color': 'green', 'ls': '--'}
            },
            "MACD": {
                'macd_hist': {'color': 'blue', 'type': 'bar'}, # MACD Histogram as bars
                # If you calculate macd_line and macd_signal:
                # 'macd_line': {'color': 'blue'},
                # 'macd_signal': {'color': 'orange'}
            },
            "Stochastic": {
                'slow_k': {'color': 'blue'},
                'slow_d': {'color': 'orange'}
            },
            "Volume": {
                'volume': {'color': 'gray', 'type': 'bar'}, # Volume as bars
                # Optionally plot avg volume or spikes if calculated
                'vol_avg': {'color': 'red'}
            },
            "ATR": {
                'atr': {'color': 'red'},
                # Optionally plot avg ATR
                'avg_atr': {'color': 'orange'}
            },
            "NWE":{
                'nwe_upper': {'color': 'pink'},
                'nwe_center': {'color': 'red'},
                'nwe_lower': {'color': 'brown'}
            }
            # Add more subplots for other indicators as needed
            # "OBV": {
            #     'obv': {'color': 'purple'}
            # },
        }
    }
    nwe_h = 8 # 16 # Bandwidth (h) - Controls smoothness/sensitivity
    nwe_sigma = 0.0 # Shape parameter (sigma) - Often kept at 0 or small values
    nwe_r = 0.0 # Position parameter (r) - Often kept at 0 or small values
    nwe_atr_mult = 3.0 # Multiplier for ATR to set envelope width
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.max_stake_percentage = 0.04  # 2% of wallet per trade
        self.max_stake_cap = 500.0  # Cap stake at 1500 USDT
        self.fallback_balance = 1000.0  # Fallback for backtesting


    def nadaraya_watson_estimator(self, src: np.ndarray, h: float, sigma: float, r: float) -> np.ndarray:
        """
        Calculates the Nadaraya-Watson estimator using vectorized operations for improved performance.
        src: Source data (e.g., closing prices) - 1D NumPy array of shape (n,)
        h: Bandwidth parameter
        sigma: Shape parameter (often 0)
        r: Position parameter (often 0)
        Returns: Estimated values - 1D NumPy array of shape (n,)
        """
        n = len(src)
        if n == 0:
            return np.array([])

        # Create index arrays for i and j
        # i_indices shape: (n, 1)
        i_indices = np.arange(n).reshape(-1, 1)
        # j_indices shape: (1, n)
        j_indices = np.arange(n).reshape(1, -1)

        # Calculate the normalized distance matrix Z
        # Z[i, j] = (i - j) / h
        # Broadcasting: (n, 1) - (1, n) -> (n, n)
        Z = (i_indices - j_indices) / h

        # Calculate the Gaussian kernel matrix K
        # K[i, j] = exp(-0.5 * (Z[i, j])^2)
        # This is equivalent to the Gaussian part of the Epanechnikov kernel simplification
        K = np.exp(-0.5 * np.power(Z, 2))

        # If sigma and r are used in a more complex way (beyond standard NW), they would be applied here.
        # For now, assuming they are part of the base kernel or kept at 0 as often seen.
        # W is the weight matrix
        W = K # Simplified weight calculation as in the original loop

        # Handle potential division by zero sum of weights
        # Sum of weights for each i (sum over j)
        # sum_W shape: (n,)
        sum_W = np.sum(W, axis=1)

        # Calculate the numerator: sum of (W[i, j] * src[j]) for each i
        # This is a dot product: W (n x n) * src (n x 1) -> result (n x 1)
        # src_for_mult needs to be (1, n) to broadcast correctly with W (n, n) for element-wise mult
        src_for_mult = src.reshape(1, -1) # Shape (1, n)
        # Element-wise multiplication of weights and source values
        W_times_src = W * src_for_mult # Broadcasting (n, n) * (1, n) -> (n, n)
        # Sum along the j axis (axis=1) to get the numerator for each i
        sum_W_times_src = np.sum(W_times_src, axis=1) # Shape (n,)

        # Avoid division by zero and calculate the estimate
        # Use np.where to handle cases where sum_W is zero
        # If sum_W[i] is 0, use src[i] as fallback
        y = np.where(sum_W > 0, sum_W_times_src / sum_W, src)

        return y
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 3.0  # Fixed 3x leverage for futures

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return False
        last_candle = dataframe.iloc[-1]
        if 'side' in kwargs and kwargs['side'] == 'short':
            return last_candle['enter_short']
        return last_candle['enter_long']

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        # Block toxic exits
        # return True
        return exit_reason != 'exit_signal'

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        # Dynamic stake based on ATR
        if 'current_balance' in kwargs:
            balance = kwargs['current_balance']
        else:
            try:
                balance = self.wallets.get_free(self.config['stake_currency'])
            except Exception as e:
                self.logger.error(f"Error retrieving balance for {pair}: {str(e)}, using fallback")
                balance = self.fallback_balance

        if balance <= 0:
            balance = self.fallback_balance

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        max_trades = int(balance / self.max_stake_cap)
        self.max_open_trades = max(3, min(6, max_trades))  # Dynamic scaling  
        if len(dataframe) < 1:
            return min_stake

        # last_candle = dataframe.iloc[-1]
        # volatility = last_candle['atr'] / last_candle['close']
        stake_percentage = self.max_stake_percentage

        if dataframe['atr'].iloc[-1] > 1.5 * dataframe['avg_atr'].iloc[-1]:  
            stake_percentage = 0.03  # 3% per trade  
        else:  
            stake_percentage = 0.02  # 2% per trade  

        stake = balance * stake_percentage
        stake = min(stake, self.max_stake_cap)
        stake = max(min_stake, min(max_stake, stake))
        return stake

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     if len(dataframe) < 1:
    #         return -0.2  # Default stop loss

    #     last_candle = dataframe.iloc[-1]
    #     atr = last_candle['atr']
    #     # Dynamic stop loss based on ATR
    #     stoploss_distance = 1.5 * atr / (current_rate + 1e-8)  # Avoid division by zero
    #     return max(-stoploss_distance, -0.005)  # Minimum 0.5% stop

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands (20, 2.0 SD)
        # bbands = ta.bbands(dataframe['close'], length=20, std=2.0)
        # dataframe['bb_lower'] = bbands['BBL_20_2.0']
        # dataframe['bb_middle'] = bbands['BBM_20_2.0']
        # dataframe['bb_upper'] = bbands['BBU_20_2.0']
        # ATR (14-period) for volatility filtering
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        dataframe['avg_atr'] = ta.sma(dataframe['atr'], length=20)

        # --- Nadaraya-Watson Envelope Calculation ---
        # Using closing prices as the source
        src = dataframe['close'].values

        # Calculate the Nadaraya-Watson Estimator (Center Line)
        nw_estimator = self.nadaraya_watson_estimator(src, self.nwe_h, self.nwe_sigma, self.nwe_r)
        dataframe['nwe_center'] = nw_estimator

        # Calculate Upper and Lower Envelopes
        envelope_offset = self.nwe_atr_mult * dataframe['atr']
        dataframe['nwe_upper'] = dataframe['nwe_center'] + envelope_offset
        dataframe['nwe_lower'] = dataframe['nwe_center'] - envelope_offset
        # --- End NWE Calculation ---

        # Stochastic Oscillator (14-period)
        stoch = ta.stoch(dataframe['high'], dataframe['low'], dataframe['close'], k=14, d=3)
        dataframe['slow_k'] = stoch['STOCHk_14_3_3']
        dataframe['slow_d'] = stoch['STOCHd_14_3_3']

        # EMA 20 for trend filter
        dataframe['ema_20'] = ta.ema(dataframe['close'], length=20)
        dataframe['ema_50'] = ta.ema(dataframe['close'], length=50)

        # RSI (14-period)
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)

        # MACD (12, 26, 9)
        macd = ta.macd(dataframe['close'], fast=12, slow=26, signal=9)
        dataframe['macd_hist'] = macd['MACDh_12_26_9']


        # dataframe['trend_strength'] = (dataframe['ema_20'] > dataframe['ema_50'])  
        dataframe['rsi_upper'] = 70 + (dataframe['atr'] / dataframe['close']) * 100  # Dynamic overbought  
        dataframe['rsi_lower'] = 30 - (dataframe['atr'] / dataframe['close']) * 100  # Dynamic oversold  

        # Volume Spike Detection
        dataframe['vol_avg'] = ta.sma(dataframe['volume'], length=20)
        dataframe['volume_spike'] = dataframe['volume'] > 1.25 * dataframe['vol_avg']

        # Ghost Candle Filter
        candle_range = dataframe['high'] - dataframe['low']
        body_size = abs(dataframe['close'] - dataframe['open'])
        dataframe['ghost_candle'] = body_size < (0.1 * candle_range)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry Conditions
        long_conditions = [
            (dataframe['close'] <= dataframe['nwe_lower']),  # Price at lower Bollinger Band
            (dataframe['rsi'] < dataframe['rsi_lower']),  # Dynamic oversold 
            (dataframe['volume_spike']),  # Volume spike (relaxed from 1.5x)
            (dataframe['ema_20'] > dataframe['ema_50']),  # Trend filter
            ~dataframe['ghost_candle']                   # Not a ghost candle (relaxed to 20% body)
        ]
        dataframe['enter_long'] = (np.sum(long_conditions, axis=0) >= 3)  # Require 3/5 conditions

        # Short Entry Conditions
        short_conditions = [
            (dataframe['close'] >= dataframe['nwe_upper']),  # Price at upper Bollinger Band
            (dataframe['rsi'] > dataframe['rsi_upper']),  # Dynamic overbuy  
            (dataframe['volume_spike']),  # Volume spike
            (dataframe['ema_20'] < dataframe['ema_50']),  # Trend filter
            ~dataframe['ghost_candle']                   # Not a ghost candle
        ]
        dataframe['enter_short'] = (np.sum(short_conditions, axis=0) >= 3)  # Require 3/5 conditions

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Exit Conditions (2/4 required)
        long_exit_conditions = [
            (dataframe['rsi'] > dataframe['rsi_upper']),  # Dynamic overbuy  
            (dataframe['close'] < dataframe['nwe_center']),     # Price below BB middle
            (dataframe['macd_hist'] < 0),                   # MACD bearish
            (dataframe['slow_k'] < dataframe['slow_d'])      # Stochastic bearish
        ]
        dataframe['exit_long'] = (np.sum(long_exit_conditions, axis=0) >= 2)

        # Short Exit Conditions (2/4 required)
        short_exit_conditions = [
            (dataframe['rsi'] < dataframe['rsi_lower']),  # Dynamic oversold 
            (dataframe['close'] > dataframe['nwe_center']),    # Price above BB middle
            (dataframe['macd_hist'] > 0),                   # MACD bullish
            (dataframe['slow_k'] > dataframe['slow_d'])      # Stochastic bullish
        ]
        dataframe['exit_short'] = (np.sum(short_exit_conditions, axis=0) >= 2)

        return dataframe