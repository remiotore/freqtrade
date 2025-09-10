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
    process_only_new_candles = True
    stake_currency = 'USDT'
    stake_amount = 'unlimited'
    use_exit_signal = True
    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.06  
    trailing_stop_positive_offset = 0.10  
    ignore_roi_if_entry_signal = True
    use_custom_exit = True
    
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "take_profit": "limit"
    }
    
    startup_candle_count = 800
    
    minimal_roi = {
        "45": 0.05,  
        "20": 0.07,  
        "0": 0.10    
    }
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['wma_20'] = ta.WMA(dataframe, timeperiod=20)
        macd_8_21_9 = ta.MACD(dataframe, fastperiod=8, slowperiod=21, signalperiod=9)
        dataframe['macd_8_21'] = macd_8_21_9['macd']
        dataframe['macdsignal_8_21'] = macd_8_21_9['macdsignal']
        macd_12_26_9 = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd_12_26'] = macd_12_26_9['macd']
        dataframe['macdsignal_12_26'] = macd_12_26_9['macdsignal']
        macd_24_52_18 = ta.MACD(dataframe, fastperiod=24, slowperiod=52, signalperiod=18)
        dataframe['macd_24_52'] = macd_24_52_18['macd']
        dataframe['macdsignal_24_52'] = macd_24_52_18['macdsignal']
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        supertrend = pta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=7, multiplier=3.0)
        dataframe['supertrend'] = supertrend['SUPERT_7_3.0']
        dataframe['supertrend_direction'] = supertrend['SUPERTd_7_3.0']
        dataframe['supertrend_long'] = supertrend['SUPERTl_7_3.0']
        dataframe['supertrend_short'] = supertrend['SUPERTs_7_3.0']
        dataframe['adx_7'] = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=7)['ADX_7']
        dataframe['adx_14'] = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=14)['ADX_14']
        dataframe['adx_30'] = pta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=30)['ADX_30']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['vwma'] = dataframe['close'].rolling(window=10).mean()
        bbands = ta.BBANDS(dataframe, length=20)
        dataframe['BBU_20_2.0'] = bbands['upperband']  # Upper Bollinger Band
        dataframe['BBL_20_2.0'] = bbands['lowerband']  # Lower Bollinger Band
        dataframe['BBM_20_2.0'] = bbands['middleband']  # Middle Bollinger Band
        dataframe['obv'] = pta.obv(dataframe['close'], dataframe['volume'])
        dataframe['obv_change_pct'] = ((dataframe['obv'] - dataframe['obv'].shift(1)) / abs(dataframe['obv'].shift(1))) * 100
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            (dataframe['ema_50'] > dataframe['ema_100']) &  # Bullish trend
            (dataframe['ema_100'] > dataframe['ema_200']) &  # Strong long-term uptrend
            (dataframe['macd_12_26'] > dataframe['macdsignal_12_26']) &  # Bullish momentum
            (dataframe['rsi_7'] < 55) &  # Not overbought
            (dataframe['mfi'] < 45) &  # Accumulation phase
            ((dataframe['adx_14'] > 20) | (dataframe['adx_7'] > 20)) &  # Adaptive trend confirmation
            (dataframe['wma_20'] > dataframe['ema_50']) &  # Short-term trend acceleration
            (dataframe['close'] > dataframe['supertrend_long']) &  # Price above Supertrend
            (dataframe['close'] > dataframe['BBM_20_2.0']) &  # Not too late (Bollinger Band Middle)
            (dataframe['vwma'] > dataframe['vwma'].shift(1)) &  # Increasing VWMA
            (dataframe['obv'] > dataframe['obv'].shift(1))  # OBV increasing = buying pressure
        ),
        ['enter_long', 'enter_tag']
        ] = (1, 'bullish_crossover')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_50'] < dataframe['ema_100']) &
                (dataframe['macd_12_26'] < dataframe['macdsignal_12_26']) &
                (dataframe['rsi_7'] > 65) &
                (dataframe['mfi'] > 65) &
                (dataframe['close'] > dataframe['supertrend_long'] * 1.08) &
                (dataframe['vwma'] < dataframe['vwma'].shift(1)) &  # Weakening volume momentum
                (dataframe['atr'] < dataframe['atr'].rolling(14).mean() * 0.8) &
                (dataframe['adx_14'] < 20) & 
                (dataframe['close'] > dataframe['BBM_20_2.0'])
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
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        atr = last_candle["atr"]
        avg_volatility = dataframe["atr"].rolling(50).mean().iloc[-1]
        adx = last_candle["adx_14"]

        # Default ROI Target
        roi_target = 0.10  

        # Dynamic ROI Logic
        if atr > avg_volatility * 1.2 and adx_14 > 30:  
            roi_target = 0.20  # High volatility + Strong trend → Hold longer
        elif atr < avg_volatility * 0.8 and adx_14 < 20:  
            roi_target = 0.05  # Low volatility + Weak trend → Exit early

        return current_profit >= roi_target

    # Callback: Custom Stoploss
    use_custom_stoploss = True
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                current_rate: float, current_profit: float, after_fill: bool, 
                **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe.iloc[-1]["atr"]
        adx_30 = dataframe.iloc[-1]["adx_30"]

        # ATR-Based Dynamic Stop
        dynamic_stop = max(-2.0 * atr, -0.10)  

        # ADX-Based Stop (Strong trends allow more room)
        adx_stop = max(-0.5 * atr, -adx_30 / 100)

        # Adaptive Stop-Loss Logic
        return min(dynamic_stop, adx_stop) 

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