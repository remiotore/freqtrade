from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)


class SOMY(IStrategy):
    """
    SOMY Strategy with Safer, Confluence-Based Entries 
    """
    # Set a default timeframe
    timeframe = '5m'

    # --- 1. Define Base Unleveraged Parameters ---
    base_stoploss = -0.15
    base_minimal_roi = {
        "0": 0.12,
        "60": 0.05,
        "180": 0.02,
        "360": 0
    }
    base_trailing_offset = 0.05
    base_trailing_positive = 0.02

    # --- 2. DYNAMIC INITIALIZATION ---
    def __init__(self, config: dict):
        super().__init__(config)
        trading_mode = self.config.get('trading_mode', 'spot')
        leverage = self.config.get('leverage', 1.0)

        if trading_mode == 'futures' and leverage > 1:
            self.leverage = leverage
            self.stoploss = self.base_stoploss / self.leverage
            self.trailing_stop_positive_offset = self.base_trailing_offset / self.leverage
            self.trailing_stop_positive = self.base_trailing_positive / self.leverage
            self.minimal_roi = {str(k): v / self.leverage for k, v in self.base_minimal_roi.items()}
        else:
            self.leverage = 1.0
            self.stoploss = self.base_stoploss
            self.minimal_roi = self.base_minimal_roi
            self.trailing_stop_positive_offset = self.base_trailing_offset
            self.trailing_stop_positive = self.base_trailing_positive

    # --- 3. Static Strategy Properties ---
    process_only_new_candles = True
    startup_candle_count = 300
    trailing_stop = True
    trailing_only_offset_is_reached = True

    # --- Buy/sell parameters ---
    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)
    buy_24h_min_pct = DecimalParameter(-30.0, 0.0, default=-24.3, decimals=1, space='buy', optimize=True)
    buy_24h_max_pct = DecimalParameter(0.0, 200.0, default=24.3, decimals=1, space='buy', optimize=True)
    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)

    # --- 4. Protections ---
    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 96},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 144, "trade_limit": 20,
                "stop_duration_candles": 12, "max_allowed_drawdown": 0.15
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24, "trade_limit": 3,
                "stop_duration_candles": 12, "only_per_pair": False
            }
        ]

    # --- 5. Indicator Population ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Original Entry indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['24h_change_pct'] = (dataframe['close'].pct_change(periods=288) * 100)

        # Trend Indicators
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50) # Changed ftt.sma to ta.SMA
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200) # Changed ftt.sma to ta.SMA

        # Confluence Indicators for Safer Entries
        dataframe['adx'] = ta.ADX(dataframe) # Changed ftt.adx to ta.ADX
        macd = ta.MACD(dataframe) # Changed ftt.macd to ta.MACD
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Bollinger Bands: corrected ftt.bollinger_bands to ta.BBANDS and variable name 'bollinger' to 'Bollinger'
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_lowerband'] = bollinger['lowerband'] # Use 'lowerband' for TA-Lib output
        dataframe['bb_upperband'] = bollinger['upperband'] # Use 'upperband' for TA-Lib output

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14) # Changed ftt.atr to ta.ATR
        dataframe['atr_sma'] = ta.SMA(dataframe['atr'], timeperiod=20) # Changed ftt.sma to ta.SMA

        # Stochastic RSI: Using pandas_ta which is already imported as pta
        # Parameters for pta.stochrsi are typically close, rsi_length, k, d
        stoch_rsi = pta.stochrsi(close=dataframe['close'], rsi_length=14, k=3, d=3)
        # pandas_ta returns columns like 'STOCHRSIk_14_3_3'
        dataframe['stoch_rsi_k'] = stoch_rsi[f'STOCHRSIk_14_3_3']

        # Williams %R
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)

        # Add longer-term EMA for trend confirmation
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50) # Changed ftt.ema to ta.EMA
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100) # Changed ftt.ema to ta.EMA

        # Exit indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        
        return dataframe

    # --- 6. Entry Logic ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Original buy conditions
        buy_conditions_1 = (
            (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
            (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
            (dataframe['rsi'] > self.buy_rsi_32.value) &
            (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
            (dataframe['cti'] < self.buy_cti_32.value) &
            (dataframe['24h_change_pct'] > self.buy_24h_min_pct.value) &
            (dataframe['24h_change_pct'] < self.buy_24h_max_pct.value)
        )
        dataframe.loc[buy_conditions_1, ['enter_long', 'enter_tag']] = (1, 'buy_original')

        # --- SAFER, CONFLUENCE-BASED PULLBACK ENTRIES ---
        safe_long_conditions = (
            (dataframe['sma_50'] > dataframe['sma_200']) &
            (dataframe['adx'] > 25) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['close'] < dataframe['bb_lowerband'])
        )
        dataframe.loc[safe_long_conditions, ['enter_long', 'enter_tag']] = (1, 'safe_pullback_long')

        safe_short_conditions = (
            (dataframe['sma_50'] < dataframe['sma_200']) &
            (dataframe['adx'] > 25) &
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['close'] > dataframe['bb_upperband'])
        )
        dataframe.loc[safe_short_conditions, ['enter_short', 'enter_tag']] = (1, 'safe_pullback_short')

        # --- NEW: VOLATILITY-CONFIRMED BREAKOUT ENTRIES ---
        
        # ATR Breakout Long: Bullish trend + Break of recent high + High volatility
        atr_breakout_long_conditions = (
            (dataframe['sma_50'] > dataframe['sma_200']) &  # 1. Bullish Trend
            (dataframe['close'] > dataframe['high'].shift(1).rolling(20).max()) & # 2. Breakout of 20-candle high
            (dataframe['atr'] > dataframe['atr_sma'] * 1.25) # 3. Volatility is 25% above average
        )
        dataframe.loc[atr_breakout_long_conditions, ['enter_long', 'enter_tag']] = (1, 'atr_breakout_long')

        # ATR Breakout Short: Bearish trend + Break of recent low + High volatility
        atr_breakout_short_conditions = (
            (dataframe['sma_50'] < dataframe['sma_200']) &  # 1. Bearish Trend
            (dataframe['close'] < dataframe['low'].shift(1).rolling(20).min()) & # 2. Breakout of 20-candle low
            (dataframe['atr'] > dataframe['atr_sma'] * 1.25) # 3. Volatility is 25% above average
        )
        dataframe.loc[atr_breakout_short_conditions, ['enter_short', 'enter_tag']] = (1, 'atr_breakout_short')

        # DIP LONG ENTRY: Bullish trend + Pullback to MA + Oversold Oscillator
        dip_long_conditions = (
            # 1. Trend Confirmation: EMA 50 is above EMA 100
            (dataframe['ema_50'] > dataframe['ema_100']) &
            
            # 2. Pullback Confirmation: Price closes near or below the middle Bollinger Band
            (dataframe['close'] < dataframe['bb_lowerband'].shift(1)) &

            # 3. Oscillator Confirmation: Williams %R is oversold
            (dataframe['willr'] < -75) &

            # 4. Momentum Confirmation: MACD histogram is starting to turn up
            (dataframe['macd'] > dataframe['macdsignal'])
        )
        dataframe.loc[dip_long_conditions, ['enter_long', 'enter_tag']] = (1, 'dip_long')

        # DIP SHORT ENTRY: Bearish trend + Rally to MA + Overbought Oscillator
        dip_short_conditions = (
            # 1. Trend Confirmation: EMA 50 is below EMA 100
            (dataframe['ema_50'] < dataframe['ema_100']) &

            # 2. Pullback Confirmation: Price closes near or above the middle Bollinger Band
            (dataframe['close'] > dataframe['bb_upperband'].shift(1)) &

            # 3. Oscillator Confirmation: Williams %R is overbought
            (dataframe['willr'] > -25) &
            
            # 4. Momentum Confirmation: MACD histogram is starting to turn down
            (dataframe['macd'] < dataframe['macdsignal'])
        )
        dataframe.loc[dip_short_conditions, ['enter_short', 'enter_tag']] = (1, 'dip_short')

        return dataframe

    # --- 7. Exit Logic ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['fastk'] > self.sell_fastx.value), ['exit_long', 'exit_tag']] = (1, 'exit_long_fastk')
        dataframe.loc[(dataframe['fastk'] < (100 - self.sell_fastx.value)), ['exit_short', 'exit_tag']] = (1, 'exit_short_fastk')
        return dataframe

    # --- 8. Custom Stoploss ---
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if current_time - timedelta(hours=10) > trade.open_date_utc and current_profit > -0.10:
            return 0.001
        if current_time - timedelta(hours=7) > trade.open_date_utc and current_profit > -0.05:
            return 0.001
        return 1.0

    # --- 9. Custom Exit Signal ---
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, dataframe: DataFrame, **kwargs):
        last_candle = dataframe.iloc[-1].squeeze()
        if current_profit > -0.03:
            if trade.is_short and last_candle["cci"] < -80:
                return "cci_exit_short"
            if not trade.is_short and last_candle["cci"] > 80:
                return "cci_exit_long"
        return None
