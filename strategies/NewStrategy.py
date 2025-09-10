from pandas_ta import trend
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import pandas as pd
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
import pandas_ta as pta

class NewStrategy(IStrategy):
    
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    stake_currency = 'USDT'
    stake_amount = 'unlimited'
    use_exit_signal = True
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.10  # 0.5% profit lock
    trailing_stop_positive_offset = 0.15  # Offset to activate trailing stop
    
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "take_profit": "market"
    }
    
    startup_candle_count = 30
    
    minimal_roi = {
        "120": 0.07,  
        "60": 0.05,  
        "0": 0.12    
    }
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        supertrend = pta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=7, multiplier=3.0)
        dataframe['supertrend'] = supertrend['SUPERT_7_3.0']
        dataframe['supertrend_direction'] = supertrend['SUPERTd_7_3.0']
        dataframe['supertrend_long'] = supertrend['SUPERTl_7_3.0']
        dataframe['supertrend_short'] = supertrend['SUPERTs_7_3.0']
        adx = pta.adx(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=14)
        dataframe['adx'] = adx['ADX_14']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['vwma'] = dataframe['close'].rolling(window=10).mean()
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_50'] > dataframe['ema_100']) & 
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['rsi'] < 55) &  # Increased from 40
                (dataframe['mfi'] < 45) &  # Increased from 30
                #(dataframe['adx'] > 20) &  # Strong trend filter
                #(dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2) &  # Confirm strong volume
                (dataframe['close'] > dataframe['supertrend_long']) & 
                (dataframe['vwma'] > dataframe['vwma'].shift(1)) &
                (dataframe['atr'] > dataframe['atr'].rolling(14).mean() * 1.1)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'bullish_crossover')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_50'] < dataframe['ema_100']) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['rsi'] > 65) &
                (dataframe['mfi'] > 65) &
                #(dataframe['adx'] < 20) &  # Strong trend filter
                #(dataframe['volume'] < dataframe['volume'].rolling(20).mean() * 0.9) &  # Confirm low volume exit
                (dataframe['close'] > dataframe['supertrend_long'] * 1.08) &
                (dataframe['vwma'] < dataframe['vwma'].shift(1)) &  # Weakening volume momentum
                (dataframe['atr'] < dataframe['atr'].rolling(14).mean() * 0.8)  # Lower volatility signaling exit
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'bearish_reversal')
        # **New Adaptive Trailing Stop-Loss** - Only exit if close drops below the adjusted ATR-based trailing stop
        dataframe['trailing_stop'] = dataframe['close'] - (dataframe['atr'] * 2.0)
        dataframe.loc[
            (dataframe['close'] < dataframe['trailing_stop']),
            ['exit_long', 'exit_tag']
        ] = (1, 'trailing_stop_loss')
        
        return dataframe
    
    # Callback: Custom Stoploss
    use_custom_stoploss = True
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, after_fill: bool, 
                    **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe.iloc[-1]["atr"]
        adx = dataframe.iloc[-1]["adx"]

        # If trend weakens (ADX < 15), exit quickly
        if adx < 15 and current_profit < -0.02:
            return -0.02  # Cut losses early
    
        # ATR-Based Dynamic Stop
        dynamic_stop = max(-2.0 * atr, -0.10)  

        # Time-Based Stop for Early Loss Cuts
        trade_duration = (current_time - trade.open_date_utc).days
        if trade_duration > 3 and current_profit < -0.02:  
            return max(-0.4 * atr, current_profit)  # Reduce exposure on losers

        # Secure Partial Profits
        if current_profit >= 0.03:  
            return max(0.00, current_profit * 0.5)  

        return dynamic_stop  # Default ATR stop-loss


    # Callback: Confirm Trade Entry
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        return True  # Always allow trade entry
    
    # Callback: Confirm Trade Exit
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        return True  # Always allow trade exit
    
    # Callback: Order Filled
    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime, **kwargs) -> None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade.set_custom_data(key="entry_candle_high", value=last_candle["high"])

        # Time-Based Stop for Early Loss Cuts
        trade_duration = (current_time - trade.open_date_utc).days
        if trade_duration > 4 and current_profit < 0.00:  
            return max(-0.5 * atr, current_profit)  # Use dynamic ATR stop on losers

        # Break-Even Stop (Secures Small Gains)
        if current_profit >= 0.02:  # Adjusted from 4% to 3.5%
            return max(0.00, current_profit * 0.5)  # Move SL to secure partial profit  

        return dynamic_stop  # Default ATR-based stop


    # Callback: Confirm Trade Entry
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                        time_in_force: str, current_time: datetime, entry_tag: str | None,
                        side: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # Avoid illiquid or highly volatile trades
        if last_candle["atr"] > last_candle["close"] * 0.05:  # 5% ATR is too volatile
            return False
    
        return True  # Only enter if conditions are favorable

    
    # Callback: Confirm Trade Exit
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        return True  # Always allow trade exit
    
    # Callback: Order Filled
    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime, **kwargs) -> None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade.set_custom_data(key="entry_candle_high", value=last_candle["high"])