import talib.abstract as ta
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class RSIBB_V2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    use_exit_signal = False
    
    minimal_roi = {}
    stoploss = -0.01
    trailing_stop = False
    
    max_open_trades = 1
    
    @property
    def plot_config(self):
        return {
            'main_plot': {
                'bbu' : { 'color' : 'blue' },
                'bbm' : { 'color' : 'orange' },
                'bbl' : { 'color' : 'blue' },
                'sma' : { 'color' : 'blue' },
            },
            'subplots': {
                "RSI" : {
                    "rsi_fast" : { 'color' : 'yellow' },
                    "rsi_slow" : { 'color' : 'red' },
                }
            }
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sma"] = ta.SMA(dataframe["close"], timeperiod=20)
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=6)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=12)
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bbl'] = bollinger['lower']
        dataframe['bbm'] = bollinger['mid']
        dataframe['bbu'] = bollinger['upper']
        dataframe['bbw'] = dataframe['bbu'] - dataframe['bbl']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi_fast"], 70)) &
                (qtpylib.crossed_above(dataframe["close"], dataframe['bbu'])) &
                (dataframe["bbw"].diff() > 0) &
                (dataframe["rsi_fast"].diff() > 0) &
                (dataframe["rsi_slow"].diff() > 0)
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, 
                    current_rate: float, current_profit: float, **kwargs) -> float:

        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if df.empty or 'atr' not in df.columns:
            return -0.10

        atr = df['atr'].iloc[-1]

        if pd.isna(atr) or atr <= 0:
            return -0.10

        stoploss = -max(0.04, min(atr / current_rate, 0.12))
        return stoploss

    def leverage(self, pair: str, current_time: datetime, current_rate: float, 
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if len(df) < self.leverage_window_size.value:
            return proposed_leverage
        
        last_row = df.iloc[-1]
        rsi = last_row.get("rsi_slow", 50)
        atr = last_row.get("atr", 0)
        sma = last_row.get("sma", current_rate)
    
        lev = self.leverage_base.value

        if side == "long":

            if rsi < self.leverage_rsi_low.value:
                lev *= self.leverage_long_increase_factor.value

            elif rsi > self.leverage_rsi_high.value:
                lev *= self.leverage_long_decrease_factor.value
                
            if atr > 0 and current_rate > 0:
                volatility_ratio = atr / current_rate
                if volatility_ratio > self.leverage_atr_threshold_pct.value:
                    lev *= self.leverage_volatility_decrease_factor.value
                    
            if current_rate < sma:
                lev *= self.leverage_long_decrease_factor.value

        return round(max(1, min(lev, max_leverage)), 2)
