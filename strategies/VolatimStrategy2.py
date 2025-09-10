from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
import numpy as np
from datetime import datetime
from freqtrade.persistence import Trade

class VolatimStrategyOptimized(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # Optimized ROI
    minimal_roi = {
        "0": 0.08,    # 8% for immediate exit
        "15": 0.05,   # 5% after 15 minutes
        "30": 0.03,   # 3% after 30 minutes
        "60": 0.01    # 1% after 60 minutes
    }
    
    # Dynamic stoploss
    stoploss = -0.10  # Initial stoploss (will be overridden by custom_stoploss)
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    
    # Risk management
    startup_candle_count = 50
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    
    # Protections
    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 10,
                "stop_duration_candles": 72,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 144,
                "trade_limit": 4,
                "stop_duration_candles": 36,
                "only_per_pair": True
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI with multiple timeframes
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        dataframe['rsi_7'] = ta.rsi(dataframe['close'], length=7)
        dataframe['rsi_21'] = ta.rsi(dataframe['close'], length=21)
        
        # Price extremes
        dataframe['rolling_max'] = dataframe['close'].rolling(window=20).max()
        dataframe['rolling_min'] = dataframe['close'].rolling(window=20).min()
        dataframe['local_max'] = (dataframe['close'] >= dataframe['rolling_max'].shift(1))
        dataframe['local_min'] = (dataframe['close'] <= dataframe['rolling_min'].shift(1))
        
        # Returns analysis
        dataframe['returns'] = dataframe['close'].pct_change()
        dataframe['returns_roll_mean'] = dataframe['returns'].rolling(window=20).mean()
        dataframe['returns_roll_mean_cumsum'] = dataframe['returns_roll_mean'].cumsum()
        
        # Volatility bands (dynamic multiplier)
        std_multiplier = 1.8  # Optimized from backtesting
        dataframe['returns_roll_std'] = dataframe['returns'].rolling(window=20).std()
        dataframe['returns_roll_mean_cumsum_upper'] = (
            dataframe['returns_roll_mean_cumsum'] + std_multiplier * dataframe['returns_roll_std']
        )
        dataframe['returns_roll_mean_cumsum_lower'] = (
            dataframe['returns_roll_mean_cumsum'] - std_multiplier * dataframe['returns_roll_std']
        )
        
        # BTC correlation
        btc_pair = 'BTC/USDT'
        btc_dataframe = self.dp.get_pair_dataframe(pair=btc_pair, timeframe=self.timeframe)
        btc_dataframe['btc_returns'] = btc_dataframe['close'].pct_change()
        btc_dataframe['btc_returns_roll_mean'] = btc_dataframe['btc_returns'].rolling(window=20).mean()
        btc_dataframe['btc_returns_roll_mean_cumsum'] = btc_dataframe['btc_returns_roll_mean'].cumsum()
        dataframe = dataframe.join(btc_dataframe[['btc_returns', 'btc_returns_roll_mean', 'btc_returns_roll_mean_cumsum']])
        
        # Volume analysis
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # Trend indicators
        dataframe['ema50'] = ta.ema(dataframe['close'], length=50)
        dataframe['ema200'] = ta.ema(dataframe['close'], length=200)
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        
        # Momentum
        dataframe['mom'] = ta.mom(dataframe['close'], length=5)
        dataframe['macd'] = ta.macd(dataframe['close']).iloc[:, 0]
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (
                ((dataframe['rsi'] < 30) | 
                 ((dataframe['rsi_7'] < 25) & (dataframe['rsi'] < 35))) &  # Multi-timeframe RSI confirmation
                (dataframe['close'] < dataframe['returns_roll_mean_cumsum_lower'] * 0.98) &  # Conservative entry threshold
                (dataframe['volume_ratio'] > 1.2) &  # Strong volume confirmation
                (dataframe['local_min'] == True) &  # Local minimum
                (dataframe['btc_returns_roll_mean_cumsum'] > 0) &  # BTC in positive trend
                (dataframe['close'] > dataframe['close'].shift(3)) &  # Recent upward momentum
                (dataframe['ema50'] > dataframe['ema200']) &  # Price above EMAs
                (dataframe['mom'] > 0) &  # Positive momentum
                (dataframe['macd'] > 0)  # MACD signal line crossover
            ),
            (
                (dataframe['rsi_21'] < 30) &  # Longer-term RSI oversold
                (dataframe['close'] < dataframe['ema50'] * 0.97) &  # Significant discount to EMA50
                (dataframe['volume_ratio'] > 1.5) &  # Very strong volume
                (dataframe['atr'] > dataframe['atr'].rolling(50).mean())  # High volatility
            )
        ]
        
        dataframe.loc[reduce(lambda x, y: x | y, conditions), ['buy', 'buy_tag']] = (1, 'volatility_entry')
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe['rsi'] > 70) |
            (dataframe['rsi_7'] > 75) |
            (dataframe['close'] > dataframe['returns_roll_mean_cumsum_upper'] * 1.02) |
            ((dataframe['local_max'] == True) & (dataframe['volume_ratio'] < 0.8)) |
            (dataframe['returns_roll_mean_cumsum'] < -0.8) |
            (dataframe['close'] < dataframe['ema50'] * 0.98)  # Dynamic exit threshold
        ]
        
        dataframe.loc[reduce(lambda x, y: x | y, conditions), ['sell', 'exit_tag']] = (1, 'volatility_exit')
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, 
                       current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Base stoploss at 10% or 1.5*ATR, whichever is smaller
        atr_stoploss = last_candle['atr'] * 1.5 / current_rate
        return min(-0.10, -atr_stoploss)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, 
                          proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Dynamic position sizing based on volatility
        atr_ratio = last_candle['atr'] / current_rate
        risk_factor = 0.01  # Risk 1% of capital per trade
        
        # Adjust for current portfolio performance
        if self.wallets.get_total_profit() < 0:
            risk_factor *= 0.8  # Reduce risk during drawdown
            
        stake_amount = self.wallets.get_total_stake_amount() * risk_factor / (atr_ratio * 2)
        
        return min(max(stake_amount, min_stake), max_stake)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                         rate: float, time_in_force: str, exit_reason: str,
                         current_time: datetime, **kwargs) -> bool:
        # Allow partial exits
        if exit_reason == 'volatility_exit' and trade.open_rate * 1.05 < rate:
            return True
            
        # Don't exit if we're in strong momentum
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if (last_candle['mom'] > last_candle['mom'].rolling(5).mean() and 
            last_candle['volume_ratio'] > 1.5):
            return False
            
        return True