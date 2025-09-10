import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from functools import reduce
from freqtrade.strategy import IStrategy, informative
from freqtrade.exchange import timeframe_to_minutes
from datetime import datetime, timedelta
from typing import Dict, List

class SteveLava(IStrategy):
    """
    This strategy is optimized based on the specified indicators with the best performance parameters
    """
    # Strategy interface version - allow new iterations of the strategy interface
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.20,
        "30": 0.10,
        "60": 0.05,
        "120": 0.025
    }

    # Stoploss
    stoploss = -0.15  # Tightened from -0.3 for better capital preservation

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Optimal ticker interval for the strategy
    timeframe = '5m'
    
    # Run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Informative timeframe
    inf_1h = '1h'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # BTC threshold configuration
    btc_threshold = -0.03  # BTC allowed drop before preventing new entries

    # EMA periods
    ema_slow_period = 50
    ema_fast_period = 20
    ema_100_period = 100

    # RSI periods
    rsi_period = 14
    rsi_fast_period = 4
    rsi_slow_period = 20
    rsi_1h_period = 14

    # Volume means
    volume_mean_12_period = 12
    volume_mean_24_period = 24

    # Other indicators
    cti_period = 40
    ewo_fast_period = 12
    ewo_slow_period = 26
    lookback_candles = 4
    vwap_period = 20
    
    # Buy params (optimized for best performance)
    buy_rsi_fast_threshold = 35
    buy_rsi_threshold = 30
    buy_rsi_1h_threshold = 60
    buy_ewo_high = 2.0
    buy_ewo_low = -6.0
    

    @property
    def plot_config(self):
        return {
            'main_plot': {
                'ema_slow': {'color': 'blue'},
                'ema_100': {'color': 'green'},
                'vwap': {'color': 'orange'},
                'vwap_upperband': {'color': 'red'},
            },
            'subplots': {
                "RSI": {
                    'rsi': {'color': 'purple'},
                    'rsi_fast': {'color': 'blue'},
                    'rsi_slow': {'color': 'green'},
                    'rsi_1h': {'color': 'red'},
                },
                "EWO": {
                    'ewo': {'color': 'orange'},
                },
                "CTI": {
                    'cti_40_1h': {'color': 'red'},
                },
                "VOL": {
                    'volume': {},
                    'volume_mean_12': {'color': 'blue'},
                    'volume_mean_24': {'color': 'green'},
                },
            }
        }

    def informative_pairs(self):
        # Don't use informative pairs during backtesting to avoid errors
        if not self.dp or self.dp.runmode.value in ('backtest', 'hyperopt'):
            return []
            
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_1h) for pair in pairs]
        
        # Add BTC pair for market correlation
        if "BTC/USDT" not in pairs:
            informative_pairs.append(("BTC/USDT", self.timeframe))
            informative_pairs.append(("BTC/USDT", self.inf_1h))
            informative_pairs.append(("BTC/USDT", '1d'))
        
        return informative_pairs

    def get_btc_info(self, dataframe: DataFrame) -> DataFrame:
        # Set default values for backtesting
        if not self.dp or not self.dp.runmode.value in ('live', 'dry_run'):
            dataframe['btc_5m_1d_diff'] = 0
            dataframe['btc_ema_fast'] = 0
            return dataframe
        
        try:
            btc_tf = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
            btc_1d_tf = self.dp.get_pair_dataframe("BTC/USDT", '1d')
            
            if btc_tf is not None and btc_1d_tf is not None and not btc_tf.empty and not btc_1d_tf.empty:
                # Get the BTC 5m vs 1d price difference
                dataframe['btc_5m_1d_diff'] = 100 * (btc_tf['close'].iloc[-1] - btc_1d_tf['open'].iloc[-1]) / btc_1d_tf['open'].iloc[-1]
                dataframe['btc_ema_fast'] = ta.EMA(btc_tf, timeperiod=self.ema_fast_period).iloc[-1]
            else:
                dataframe['btc_5m_1d_diff'] = 0
                dataframe['btc_ema_fast'] = 0
        except Exception:
            dataframe['btc_5m_1d_diff'] = 0
            dataframe['btc_ema_fast'] = 0
        
        return dataframe

    def normalize(self, data, min_value, max_value):
        normalized = (data - min_value) / (max_value - min_value)
        return normalized

    def heikin_ashi(self, dataframe):
        """
        Calculate Heikin-Ashi candles manually 
        """
        # Create a new dataframe
        heikin_ashi = pd.DataFrame(index=dataframe.index)
        
        # Calculate ha_open - first ha_open is the average of first open and close
        ha_open = pd.Series(index=dataframe.index)
        ha_open.iloc[0] = (dataframe['open'].iloc[0] + dataframe['close'].iloc[0]) / 2
        for i in range(1, len(dataframe)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + dataframe['close'].iloc[i-1]) / 2
        
        # Calculate ha_close
        ha_close = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        
        # Calculate ha_high and ha_low
        ha_high = dataframe['high'].copy()
        ha_low = dataframe['low'].copy()
        
        for i in range(len(dataframe)):
            ha_high.iloc[i] = max(dataframe['high'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
            ha_low.iloc[i] = min(dataframe['low'].iloc[i], ha_open.iloc[i], ha_close.iloc[i])
        
        # Assign the calculated values to the new dataframe
        heikin_ashi['ha_open'] = ha_open
        heikin_ashi['ha_close'] = ha_close
        heikin_ashi['ha_high'] = ha_high
        heikin_ashi['ha_low'] = ha_low
        
        return heikin_ashi

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate all indicators used by the strategy
        """
        # Basic indicators - RSIs are top performing based on metrics
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_period)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=self.ema_100_period)
        
        # RSI indicators (key performance drivers)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=self.rsi_fast_period)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=self.rsi_slow_period)
        dataframe['rsi_28'] = ta.RSI(dataframe, timeperiod=28)
        dataframe['rsi_36'] = ta.RSI(dataframe, timeperiod=36)
        dataframe['rsi_42'] = ta.RSI(dataframe, timeperiod=42)
        dataframe['rsi_72'] = ta.RSI(dataframe, timeperiod=72)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)
        
        # RSI buy threshold
        dataframe['rsi_fast_buy'] = 35
        
        # Default BTC values for backtesting
        dataframe['btc_5m_1d_diff'] = 0
        dataframe['btc_ema_fast'] = 0
        
        # Volume indicators
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(window=self.volume_mean_12_period).mean()
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(window=self.volume_mean_24_period).mean()
        dataframe['relative_volume'] = dataframe['volume'] / dataframe['volume'].rolling(window=20).mean()
        
        # VWAP - using rolling_vwap to avoid lookahead bias
        dataframe['vwap'] = qtpylib.rolling_vwap(dataframe, window=self.vwap_period)
        dataframe['vwap_upperband'] = dataframe['vwap'] * 1.01
        dataframe['vwap_width'] = 0.02  # Fixed width to avoid division issues
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband2'] = bollinger['upper']
        dataframe['basic_ub'] = bollinger['upper']
        dataframe['final_ub'] = bollinger['upper']
        
        # Price offset values
        dataframe['high_offset_2'] = dataframe['high'] * 1.02
        dataframe['low_offset'] = dataframe['low'] * 0.99
        
        # EWO - Elliott Wave Oscillator
        dataframe['ewo'] = (
            ta.EMA(dataframe, timeperiod=self.ewo_fast_period) -
            ta.EMA(dataframe, timeperiod=self.ewo_slow_period)
        )
        dataframe['ewo_high'] = self.buy_ewo_high
        dataframe['ewo_low'] = self.buy_ewo_low
        
        # R_480 indicator
        dataframe['r_480'] = (dataframe['high'].rolling(480).max() - dataframe['close']) / (dataframe['high'].rolling(480).max() - dataframe['low'].rolling(480).min())
        
        # Percent change
        dataframe['pct_change_min'] = dataframe['close'].pct_change(1)
        
        # Heikin Ashi - using simplified calculation
        try:
            ha_candles = self.heikin_ashi(dataframe)
            dataframe['ha_high'] = ha_candles['ha_high']
        except:
            # Fallback if heikin ashi fails
            dataframe['ha_high'] = dataframe['high']
        
        # Pumping indicators
        dataframe['ispumping'] = (dataframe['close'] > dataframe['open'] * 1.02)
        dataframe['ispumping_rolling'] = dataframe['ispumping'].rolling(24).sum()
        dataframe['recentispumping_rolling'] = dataframe['ispumping'].rolling(8).sum()
        dataframe['isshortpumping'] = (dataframe['close'] > dataframe['open'] * 1.03)
        
        # CMF
        dataframe['cmf_div_slow'] = self.calculate_cmf(dataframe, 20)
        
        # Momentum divergence
        dataframe['momdiv_col'] = np.where(
            (dataframe['close'] > dataframe['close'].shift(1)) & 
            (dataframe['rsi'] < dataframe['rsi'].shift(1)), 
            1, 0
        )
        dataframe['momdiv_coh'] = np.where(
            (dataframe['close'] < dataframe['close'].shift(1)) & 
            (dataframe['rsi'] > dataframe['rsi'].shift(1)), 
            1, 0
        )
        
        # Trend detection
        dataframe['uptrend_1h'] = np.where(dataframe['ema_slow'] > dataframe['ema_slow'].shift(12), 1, 0)
        
        # Additional required fields
        dataframe['close_15m'] = dataframe['close']
        dataframe['ema_vwap_diff_50'] = ((dataframe['vwap'] - dataframe['ema_slow']) / dataframe['ema_slow']) * 100
        dataframe['retries'] = 0
        dataframe['adaptive'] = (dataframe['high'] + dataframe['low'] + dataframe['close'] + dataframe['open']) / 4
        dataframe['source'] = dataframe['close']
        dataframe['pm'] = 0.5
        
        # 1h timeframe indicators - default values for backtesting
        dataframe['rsi_1h'] = 50
        dataframe['cti_40_1h'] = 0
        
        # Entry parameters
        dataframe['enter_tag'] = ""
        dataframe['enter_long'] = 0
        
        return dataframe

    def calculate_cmf(self, dataframe, period):
        """Calculate Chaikin Money Flow"""
        mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
        mfv = mfv.fillna(0.0)  # float division by zero handling
        mfv *= dataframe['volume']
        cmf = mfv.rolling(period).sum() / dataframe['volume'].rolling(period).sum()
        return cmf

    def populate_informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate 1h timeframe indicators"""
        # RSI 1h
        dataframe['rsi_1h'] = ta.RSI(dataframe, timeperiod=self.rsi_1h_period)
        
        # CTI 1h - Correlation Trend Indicator
        dataframe['cti_40_1h'] = self.calculate_cti(dataframe, self.cti_period)
        
        return dataframe

    def calculate_cti(self, dataframe, period):
        """Calculate Correlation Trend Indicator"""
        return pd.Series(ta.CORREL(dataframe['close'], pd.Series(range(len(dataframe))), period), index=dataframe.index)

    def populate_informative_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Merge informative indicators into main dataframe"""
        if not self.dp:
            # Skip if DataProvider is not available
            return dataframe
            
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            # Set default values for backtesting
            dataframe['rsi_1h'] = 50
            dataframe['cti_40_1h'] = 0
            dataframe['btc_5m_1d_diff'] = 0
            return dataframe
            
        inf_1h = self.dp.get_pair_dataframe(metadata['pair'], self.inf_1h)
        
        if inf_1h is not None:
            # Join informative dataframe
            dataframe = pd.merge(
                dataframe, inf_1h[['date', 'rsi_1h', 'cti_40_1h']], 
                left_on='date', right_on='date', how='left', suffixes=('', '_1h')
            )
            
            # Fill missing values (in case of misaligned dataframes)
            dataframe['rsi_1h'] = dataframe['rsi_1h'].fillna(50)
            dataframe['cti_40_1h'] = dataframe['cti_40_1h'].fillna(0)
            
        # Add BTC correlation data if available
        if self.dp.runmode.value in ('live', 'dry_run'):
            try:
                # BTC 5m vs 1d difference
                btc_tf = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
                btc_1d_tf = self.dp.get_pair_dataframe("BTC/USDT", '1d')
                
                if btc_tf is not None and btc_1d_tf is not None and not btc_tf.empty and not btc_1d_tf.empty:
                    # Get the difference between current 5m close and daily open
                    btc_current = btc_tf['close'].iloc[-1]
                    btc_1d_open = btc_1d_tf['open'].iloc[-1]
                    
                    dataframe['btc_5m_1d_diff'] = 100 * (btc_current - btc_1d_open) / btc_1d_open
                else:
                    dataframe['btc_5m_1d_diff'] = 0
            except Exception:
                # Fallback value
                dataframe['btc_5m_1d_diff'] = 0
                
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on the most profitable indicators
        """
        dataframe.loc[:, 'enter_tag'] = ''
        conditions = []
        
        # Get top performing indicators based on provided metrics
        # RSI_112 (2.118%), RSI_84 (1.849%), RSI_72 (0.956%), RSI_42_1h (0.348%)
        
        # ENTRY CONDITION 1: RSI-based trend following
        rsi_cond = (
            (dataframe['rsi_112'] < 60) &                      # Not overbought on strongest indicator
            (dataframe['rsi_112'] > dataframe['rsi_112'].shift(1)) &  # Rising RSI
            (dataframe['rsi_84'] > 30) &                       # Not oversold on second strongest
            (dataframe['rsi_72'] > dataframe['rsi_72'].shift(1))      # Rising trend on third strongest
        )
        
        # ENTRY CONDITION 2: Volume + Trend
        volume_trend_cond = (
            (dataframe['volume'] > dataframe['volume_mean_24'] * 1.2) &  # Above average volume
            (dataframe['close'] > dataframe['ema_slow']) &               # Price above slow EMA
            (dataframe['uptrend_1h'] > 0) &                              # 1h uptrend confirmed
            (dataframe['close'] > dataframe['close'].shift(1))           # Current price rising
        )
        
        # ENTRY CONDITION 3: VWAP + EMA setup
        vwap_ema_cond = (
            (dataframe['close'] < dataframe['vwap_upperband']) &        # Not extended above VWAP
            (dataframe['ema_vwap_diff_50'] > -0.3) &                    # Price near VWAP (not too far below)
            (dataframe['vwap_width'] > 0.1) &                           # Some volatility present
            (dataframe['close'] > dataframe['ema_100'])                  # Price above EMA 100 (broader uptrend)
        )
        
        # ENTRY CONDITION 4: EWO + RSI setup (oscillator strategy)
        ewo_rsi_cond = (
            (dataframe['ewo'] > dataframe['ewo_low']) &                 # EWO above lower threshold
            (dataframe['ewo'] < dataframe['ewo_high']) &                # EWO below upper threshold
            (dataframe['rsi_fast'] < dataframe['rsi_fast_buy']) &       # RSI fast in buy zone
            (dataframe['rsi_84'] > 35) &                                # RSI not extremely oversold
            (dataframe['rsi_1h'] > 30)                                  # 1h RSI not extremely oversold
        )
        
        # ENTRY CONDITION 5: Momentum divergence
        momdiv_cond = (
            (dataframe['momdiv_col'] > 0) &                             # Bullish momentum divergence
            (dataframe['cti_40_1h'] < 0.5) &                            # 1h CTI not overbought
            (dataframe['rsi_36'] < 60) &                                # RSI not overbought
            (dataframe['close'] > dataframe['low'].shift(1))            # Current close above previous low
        )
        
        # ENTRY CONDITION 6: BTC correlation protection
        btc_cond = (
            (dataframe['btc_5m_1d_diff'] > self.btc_threshold)       
        )
        
        # ENTRY CONDITION 7: Pump protection
        pump_protection = (
            (dataframe['isshortpumping'] == False) &                    # Not currently pumping hard
            (dataframe['recentispumping_rolling'] < 3)                  # Not too many recent pumps
        )
        
        # ENTRY CONDITION 8: HMA 50 support (best performer after RSIs)
        hma_support = (
            (dataframe['close'] > dataframe['close'].rolling(50).mean()) &  # Using rolling mean instead of HMA
            (dataframe['close'].rolling(50).mean() > dataframe['close'].rolling(50).mean().shift(1))  # Rising support
        )
        
        # Combine all conditions - Entry type 1: Main strategy
        conditions.append(
            (
                rsi_cond &
                volume_trend_cond &
                vwap_ema_cond &
                btc_cond &
                pump_protection
            )
        )
        
        # Entry type 2: EWO + RSI strategy
        conditions.append(
            (
                ewo_rsi_cond &
                hma_support &
                btc_cond &
                pump_protection
            )
        )
        
        # Entry type 3: Momentum divergence strategy
        conditions.append(
            (
                momdiv_cond &
                btc_cond &
                pump_protection &
                (dataframe['rsi_84'] < 70)
            )
        )
        
        # Set tags for different entry types
        if conditions[0].sum() > 0:
            dataframe.loc[conditions[0], 'enter_tag'] = 'rsi_trend_entry'
        
        if conditions[1].sum() > 0:
            dataframe.loc[conditions[1], 'enter_tag'] = 'ewo_rsi_entry'
            
        if conditions[2].sum() > 0:
            dataframe.loc[conditions[2], 'enter_tag'] = 'momdiv_entry'
            
        # Set enter_long based on conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions) & (dataframe['volume'] > 0), 'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_tag'] = ''
        
        # Exit when RSI is overbought
        dataframe.loc[
            (
                (dataframe['rsi_112'] > 80) &                          # Strongest RSI indicator overbought
                (dataframe['rsi_84'] > 75) &                           # Confirmation from second indicator
                (dataframe['volume'] > 0)                              # Valid volume
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'rsi_overbought')
        
        # Exit on bearish momentum divergence
        dataframe.loc[
            (
                (dataframe['momdiv_coh'] > 0) &                        # Bearish momentum divergence
                (dataframe['rsi_84'] > 70) &                           # RSI high
                (dataframe['volume'] > dataframe['volume_mean_12']) &  # Above average volume
                (dataframe['close'] < dataframe['close'].shift(1))     # Price dropping
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'momdiv_exit')
        
        # Exit when price is extended too far above VWAP
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['vwap_upperband'] * 1.02) &  # Price extended above VWAP
                (dataframe['volume'] > dataframe['volume_mean_12']) &       # Above average volume
                (dataframe['rsi_84'] > 65)                                  # RSI relatively high
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'vwap_extended_exit')
        
        # Exit when EWO turns bearish with high RSI
        dataframe.loc[
            (
                (dataframe['ewo'] < -2) &                              # EWO turned bearish
                (dataframe['rsi_84'] > 60) &                           # RSI relatively high
                (dataframe['close'] < dataframe['ema_slow'])           # Price below slow EMA
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'ewo_bearish_exit')
        
        return dataframe
        
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, 
                           time_in_force: str, current_time: datetime, entry_tag: str, 
                           side: str, **kwargs) -> bool:
        # For backtesting, always return True
        if not self.dp or self.dp.runmode.value in ('backtest', 'hyperopt'):
            return True
            
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return True
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Skip trade if BTC is dropping too hard
        if 'btc_5m_1d_diff' in last_candle and last_candle['btc_5m_1d_diff'] < self.btc_threshold:
            return False
            
        # Check if we are in a pump
        if 'isshortpumping' in last_candle and last_candle['isshortpumping']:
            return False
            
        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                   current_profit: float, **kwargs):
        """
        Custom exit logic
        """
        # Skip for backtesting
        if not self.dp or self.dp.runmode.value in ('backtest', 'hyperopt'):
            return None
            
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1].squeeze()
            
            # Exit if BTC crashes hard
            if 'btc_5m_1d_diff' in last_candle and last_candle['btc_5m_1d_diff'] < -5:
                return 'btc_crash_exit'
                
            # Take profit on significant gains
            if current_profit > 0.08:
                # If RSI is high, better to exit
                if 'rsi_84' in last_candle and last_candle['rsi_84'] > 75:
                    return 'high_profit_high_rsi_exit'
        
        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: str, side: str, **kwargs) -> float:
        """
        Custom stake size based on performance
        """
        # Skip for backtesting
        if not self.dp or self.dp.runmode.value in ('backtest', 'hyperopt'):
            return proposed_stake
            
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Default to proposed stake
        stake_amount = proposed_stake
        
        # Adjust stake based on BTC volatility
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1].squeeze()
            
            # Reduce stake if BTC is volatile
            if 'btc_5m_1d_diff' in last_candle:
                btc_change = last_candle['btc_5m_1d_diff']
                
                # If BTC is dropping, reduce stake
                if btc_change < -1:
                    stake_amount = proposed_stake * 0.8
                # If BTC is rising quickly, increase stake slightly
                elif btc_change > 2:
                    stake_amount = min(proposed_stake * 1.1, max_stake)
        
        return stake_amount