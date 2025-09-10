from pandas_ta import trend
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import pandas as pd
from pandas import DataFrame
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
import pandas_ta as pta

class OptimizedStrategy(IStrategy):
    
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    process_only_new_candles = True
    stake_currency = 'USD'
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
    
    startup_candle_count = 710
    
    minimal_roi = {
        "45": 0.05,  
        "20": 0.07,  
        "0": 0.10    
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # **Trend Indicators**
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # **Momentum Indicators**
        macd_12_26_9 = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd_12_26'] = macd_12_26_9['macd']
        dataframe['macdsignal_12_26'] = macd_12_26_9['macdsignal']

        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        
        # **Volume Indicators**
        dataframe['obv'] = pta.obv(dataframe['close'], dataframe['volume'])
        dataframe['obv_change_pct'] = ((dataframe['obv'] - dataframe['obv'].shift(1)) / abs(dataframe['obv'].shift(1))) * 100
        
        # **Volatility**
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines conditions for entering a trade.
        """

        # Multi-timeframe confirmation (15m trend direction)
        higher_tf = self.dp.get_analyzed_dataframe(metadata['pair'], '15m')
        dataframe['higher_tf_trend'] = higher_tf['ema_50'] > higher_tf['ema_100']

        dataframe.loc[
            (
                (dataframe['ema_50'] > dataframe['ema_100']) &  # Bullish trend
                (dataframe['ema_100'] > dataframe['ema_200']) &  # Long-term trend confirmation
                (dataframe['macd_12_26'] > 0) &  # MACD bullish momentum
                (dataframe['rsi_7'] > 40) &  # RSI rising
                (dataframe['obv'] > dataframe['obv'].shift(1)) &  # OBV increasing
                (dataframe['atr'] < dataframe['atr'].rolling(10).mean() * 1.8) &  # ATR volatility control
                (dataframe['higher_tf_trend'] == True)  # Multi-timeframe trend confirmation
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'bullish_crossover')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines conditions for exiting a trade.
        """

        dataframe.loc[
            (
                (dataframe['obv'] < dataframe['obv'].shift(1) * 0.98) |  # OBV dropping
                (dataframe['rsi_7'] < 45) |  # RSI weakening
                (dataframe['atr'] > dataframe['atr'].rolling(10).mean() * 1.5)  # Volatility spike
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'weakening_momentum')

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe.iloc[-1]["atr"]
        
        # **Adaptive Stop-Loss Based on ATR**
        return max(-1.5 * atr, -0.08)  # Uses ATR-based SL but never worse than -8%

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        """
        Filters entries based on volatility and trend strength.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # **Avoid highly volatile or weak trades**
        if last_candle["atr"] > last_candle["close"] * 0.05:  # ATR > 5% of price is too volatile
            return False

        return True  # Only enter if conditions are favorable
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        return True  # Always allow trade exit

    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime, **kwargs) -> None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade.set_custom_data(key="entry_candle_high", value=last_candle["high"])
