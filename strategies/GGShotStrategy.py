# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class GGShotStrategy(IStrategy):
    """
    GG-SHOT AI Strategy for Freqtrade
    Pine Script'ten çevrilmiş Freqtrade versiyonu
    
    Author: GG-SHOT AI Team
    github: https://github.com/freqtrade/freqtrade-strategies
    
    How to use it?
    > freqtrade trade --strategy GGShotStrategy
    """

    INTERFACE_VERSION = 3

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.04,      # TP4: 4%
        "60": 0.025,    # TP3: 2.5% 
        "120": 0.011,   # TP2: 1.1%
        "180": 0.005,   # TP1: 0.5%
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.015  # SL: -1.5%

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # Exit signal configuration
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    # Strategy parameters - Pine Script benzeri parametreler
    # RSI Parameters
    rsi_period = IntParameter(5, 21, default=7, space="buy")
    rsi_top_limit = IntParameter(40, 60, default=45, space="buy")
    rsi_bot_limit = IntParameter(5, 20, default=10, space="buy")
    
    # Bollinger Bands Parameters
    bb_period = IntParameter(50, 200, default=100, space="buy")
    bb_std = DecimalParameter(1.0, 3.0, default=1.5, space="buy")
    
    # ATR Parameters
    atr_period = IntParameter(10, 30, default=14, space="buy")
    atr_ma_period = IntParameter(3, 10, default=5, space="buy")
    
    # Fibonacci Parameters
    sensitivity = IntParameter(100, 300, default=200, space="buy")
    
    # TP/SL Parameters
    tp1_percent = DecimalParameter(0.3, 1.0, default=0.5, space="sell")
    tp2_percent = DecimalParameter(0.8, 2.0, default=1.1, space="sell")
    tp3_percent = DecimalParameter(1.5, 4.0, default=2.5, space="sell")
    sl_percent = DecimalParameter(1.0, 3.0, default=1.5, space="sell")

    # Protection parameters
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 480,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            }
        ]

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will not be traded, but they are cached to have informative data available.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pine Script'teki indikatörlerin Freqtrade versiyonu
        Look-ahead bias olmadan hesaplama
        """
        
        # Veri kontrolü
        if dataframe.empty:
            return dataframe
            
        try:
            # RSI Calculation (Pine Script: ta.rsi(close, 7))
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
            
            # Bollinger Bands (Pine Script BB sinyali için) - qtpylib kullan
            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(dataframe), 
                window=self.bb_period.value, 
                stds=self.bb_std.value
            )
            dataframe['bb_lower'] = bollinger['lower']
            dataframe['bb_middle'] = bollinger['mid']
            dataframe['bb_upper'] = bollinger['upper']
            
            # ATR for filtering (Pine Script benzeri)
            dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
            dataframe['atr_ma'] = ta.EMA(dataframe['atr'], timeperiod=self.atr_ma_period.value)
            
            # Fibonacci Levels (Pine Script'teki sensitivity ile)
            # Rolling highest/lowest - bias-free
            dataframe['high_line'] = dataframe['high'].rolling(
                window=self.sensitivity.value, min_periods=1
            ).max()
            dataframe['low_line'] = dataframe['low'].rolling(
                window=self.sensitivity.value, min_periods=1
            ).min()
            
            # Channel range calculation
            dataframe['channel_range'] = dataframe['high_line'] - dataframe['low_line']
            
            # Fibonacci levels (Pine Script benzeri)
            dataframe['fib_236'] = dataframe['high_line'] - (dataframe['channel_range'] * 0.236)
            dataframe['fib_5'] = dataframe['high_line'] - (dataframe['channel_range'] * 0.5)  # Trend line
            dataframe['fib_786'] = dataframe['high_line'] - (dataframe['channel_range'] * 0.786)
            dataframe['fib_456'] = dataframe['high_line'] - (dataframe['channel_range'] * (-0.456))  # SL line
            
            # Pine Script'teki imba_trend_line
            dataframe['imba_trend_line'] = dataframe['fib_5']
            
            # OC2 calculation (Pine Script: oc2 = (close+open)/2)
            dataframe['oc2'] = (dataframe['close'] + dataframe['open']) / 2
            
            # Güvenli BB Signal calculation
            dataframe['bb_signal'] = 0
            
            # Veri türlerini kontrol et ve güvenli karşılaştırma yap
            try:
                # Long BB condition
                long_bb_mask = (
                    (dataframe['close'] > dataframe['bb_upper']) & 
                    (dataframe['oc2'] > dataframe['bb_upper']) &
                    ~dataframe['close'].isna() &
                    ~dataframe['bb_upper'].isna() &
                    ~dataframe['oc2'].isna()
                )
                dataframe.loc[long_bb_mask, 'bb_signal'] = 1
                
                # Short BB condition
                short_bb_mask = (
                    (dataframe['close'] < dataframe['bb_lower']) & 
                    (dataframe['oc2'] < dataframe['bb_lower']) &
                    ~dataframe['close'].isna() &
                    ~dataframe['bb_lower'].isna() &
                    ~dataframe['oc2'].isna()
                )
                dataframe.loc[short_bb_mask, 'bb_signal'] = -1
                
            except Exception as e:
                print(f"BB Signal hesaplama hatası: {e}")
                dataframe['bb_signal'] = 0
            
            # Trend Filter (Pine Script: ATR ve RSI bazlı)
            dataframe['atr_condition'] = (
                (~dataframe['atr'].isna()) & 
                (~dataframe['atr_ma'].isna()) &
                (dataframe['atr'] >= dataframe['atr_ma'])
            )
            dataframe['rsi_condition'] = (
                (~dataframe['rsi'].isna()) &
                ((dataframe['rsi'] > self.rsi_top_limit.value) | 
                 (dataframe['rsi'] < self.rsi_bot_limit.value))
            )
            dataframe['trend_filter'] = dataframe['atr_condition'] | dataframe['rsi_condition']
            
            # Pine Script can_long ve can_short mantığı
            dataframe['can_long'] = (
                (~dataframe['close'].isna()) &
                (~dataframe['imba_trend_line'].isna()) &
                (~dataframe['fib_236'].isna()) &
                (dataframe['close'] >= dataframe['imba_trend_line']) &
                (dataframe['close'] >= dataframe['fib_236']) &
                dataframe['trend_filter']
            )
            
            dataframe['can_short'] = (
                (~dataframe['close'].isna()) &
                (~dataframe['imba_trend_line'].isna()) &
                (~dataframe['fib_786'].isna()) &
                (dataframe['close'] <= dataframe['imba_trend_line']) &
                (dataframe['close'] <= dataframe['fib_786']) &
                dataframe['trend_filter']
            )
            
            # Final signals (Pine Script mantığı)
            dataframe['long_signal'] = (
                (dataframe['bb_signal'] == 1) &
                dataframe['can_long'] &
                (~dataframe['volume'].isna()) &
                (dataframe['volume'] > 0)
            )
            
            dataframe['short_signal'] = (
                (dataframe['bb_signal'] == -1) &
                dataframe['can_short'] &
                (~dataframe['volume'].isna()) &
                (dataframe['volume'] > 0)
            )
            
        except Exception as e:
            print(f"İndikatör hesaplama hatası: {e}")
            # Hata durumunda güvenli default değerler
            for col in ['rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'atr_ma', 
                       'long_signal', 'short_signal', 'bb_signal']:
                if col not in dataframe.columns:
                    if col.endswith('_signal'):
                        dataframe[col] = False
                    else:
                        dataframe[col] = np.nan
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pine Script'teki buy/sell mantığı
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        
        # Long Entry Conditions (Pine Script: buy signal)
        dataframe.loc[
            (
                dataframe['long_signal'] &
                (dataframe['volume'] > 0)  # Volume check
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'gg_long')

        # Short Entry Conditions (Pine Script: sell signal)  
        dataframe.loc[
            (
                dataframe['short_signal'] &
                (dataframe['volume'] > 0)  # Volume check
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'gg_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pine Script'teki exit mantığı
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        
        # Long Exit Conditions (ters sinyal gelirse çık)
        dataframe.loc[
            (
                dataframe['short_signal'] &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'gg_long_exit')

        # Short Exit Conditions (ters sinyal gelirse çık)
        dataframe.loc[
            (
                dataframe['long_signal'] &
                (dataframe['volume'] > 0)
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'gg_short_exit')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Pine Script'teki dynamic SL mantığı
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        """
        
        # SL percentage from parameters
        sl_ratio = -self.sl_percent.value / 100
        
        return sl_ratio

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs):
        """
        Pine Script'teki multi-TP mantığı
        Custom exit signal logic
        """
        
        # Multi-TP system (Pine Script benzeri)
        if current_profit >= (self.tp1_percent.value / 100):
            return 'tp1_hit'
        elif current_profit >= (self.tp2_percent.value / 100):
            return 'tp2_hit'  
        elif current_profit >= (self.tp3_percent.value / 100):
            return 'tp3_hit'
            
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                side: str, **kwargs) -> float:
        """
        Leverage kontrolü
        """
        return min(proposed_leverage, 10.0)  # Max 10x leverage

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str], 
                           side: str, **kwargs) -> bool:
        """
        Pine Script benzeri trade confirmation
        """
        
        # Volume check
        if 'volume' in kwargs and kwargs['volume'] <= 0:
            return False
            
        return True

    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                             current_rate: float, current_profit: float,
                             min_stake: Optional[float], max_stake: float, **kwargs):
        """
        Pine Script'teki multi-TP exit mantığı
        Position adjustment for multi-TP system
        """
        
        # Multi-TP system
        if current_profit >= (self.tp1_percent.value / 100):
            # TP1 hit - reduce position by 30%
            if not hasattr(trade, 'tp1_hit'):
                trade.tp1_hit = True
                return -(trade.amount * 0.3)
                
        elif current_profit >= (self.tp2_percent.value / 100):
            # TP2 hit - reduce position by another 30%
            if not hasattr(trade, 'tp2_hit'):
                trade.tp2_hit = True
                return -(trade.amount * 0.3)
                
        elif current_profit >= (self.tp3_percent.value / 100):
            # TP3 hit - reduce position by 50% of remaining
            if not hasattr(trade, 'tp3_hit'):
                trade.tp3_hit = True
                return -(trade.amount * 0.5)
        
        return None

    def bot_start(self, **kwargs) -> None:
        """
        Bot başlatıldığında çalışır
        """
        print("🤖 GG-SHOT AI Strategy Started!")
        print("📊 Pine Script mantığı ile Freqtrade'de çalışıyor")
        print("⚙️ Parameters:")
        print(f"   RSI Period: {self.rsi_period.value}")
        print(f"   BB Period: {self.bb_period.value}")
        print(f"   Sensitivity: {self.sensitivity.value}")
        print(f"   TP1/TP2/TP3: {self.tp1_percent.value}%/{self.tp2_percent.value}%/{self.tp3_percent.value}%")
        print(f"   SL: {self.sl_percent.value}%")

    def bot_loop_start(self, **kwargs) -> None:
        """
        Her bot loop'unda çalışır
        """
        pass

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: dict,
                           current_time: datetime, **kwargs) -> bool:
        """
        Entry order timeout check
        """
        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: dict,
                          current_time: datetime, **kwargs) -> bool:
        """
        Exit order timeout check
        """
        return False