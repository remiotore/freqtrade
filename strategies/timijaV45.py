import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

class timijaV45(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False

    minimal_roi = {
        "0": 0.02,  # Take profit at 2%
    }

    stoploss = -0.02  # Set stoploss at 2%

    use_custom_stoploss = True

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = '15m'

    process_only_new_candles = True

    max_profits = {}

    def on_trade_update(self, trade: Trade, **kwargs):
        if trade.pair not in self.max_profits:
            self.max_profits[trade.pair] = 0
        self.max_profits[trade.pair] = max(self.max_profits[trade.pair], trade.current_profit_ratio)

    def on_trade_close(self, trade: Trade, **kwargs):
        self.max_profits.pop(trade.pair, None)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        dataframe['cdlhammer'] = ta.CDLHAMMER(dataframe)
        dataframe['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(dataframe)
        dataframe['cdlengulfing'] = ta.CDLENGULFING(dataframe)

        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe.apply(pd.to_numeric, errors='coerce')

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['sma20'] > dataframe['sma50']) &
                (dataframe['sma50'] > dataframe['sma200']) &
                (dataframe['bb_percent'] < 0.3) &
                (dataframe['bb_width'] > 0.01) &
                (dataframe['macd'] > dataframe['macdsignal']) &
                (
                    (dataframe['rsi'] < 30) |
                    ((dataframe['stoch_k'] < 20) & (dataframe['stoch_d'] < 20))
                ) &
                (
                    (dataframe['cdlhammer'] == 100) |
                    (dataframe['cdlinvertedhammer'] == 100) |
                    (dataframe['cdlengulfing'] == 100)
                )
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['sma50'] < dataframe['sma200']) &
                (dataframe['bb_percent'] > 0.7) &
                (dataframe['bb_width'] < 0.02) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (
                    (dataframe['rsi'] > 70) |
                    ((dataframe['stoch_k'] > 80) & (dataframe['stoch_d'] > 80))
                ) &
                (dataframe['cdlengulfing'] == -100)
            ),
            'sell'
        ] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:
        max_profit = self.max_profits.get(pair, current_profit)
        peak_profit_drawdown = max_profit - current_profit
        atr_stoploss = self.calc_stop_loss_pct(current_rate, 6.5)

        if peak_profit_drawdown > 0.05:
            atr_stoploss *= 0.8
        elif peak_profit_drawdown > 0.1:
            atr_stoploss *= 0.6

        if current_profit > 0.005:
            atr_stoploss *= 0.8
        elif current_profit > 0.01:
            atr_stoploss *= 0.6

        return max(atr_stoploss, self.stoploss)

    def calc_stop_loss_pct(self, current_rate: float, atr_multiplier: float) -> float:
        return -atr_multiplier * current_rate

    def get_latest_candle(self, pair: str, timeframe: str, dataframe: pd.DataFrame) -> pd.Series:
        return dataframe.iloc[-1]

    def get_entry_signal(self, pair: str, timeframe: str, dataframe: pd.DataFrame) -> tuple:
        latest = self.get_latest_candle(pair, timeframe, dataframe)
        signal = 0  # Default to no signal
        enter_tag = None  # Default to no tag

        if (
            latest['sma20'] > latest['sma50'] and
            latest['sma50'] > latest['sma200'] and
            latest['bb_percent'] < 0.3 and
            latest['bb_width'] > 0.01 and
            latest['macd'] > latest['macdsignal'] and
            (
                latest['rsi'] < 30 or
                (latest['stoch_k'] < 20 and latest['stoch_d'] < 20)
            ) and
            (
                latest['cdlhammer'] == 100 or
                latest['cdlinvertedhammer'] == 100 or
                latest['cdlengulfing'] == 100
            )
        ):
            signal = 'buy'
            enter_tag = 'buy_signal'

        return signal, enter_tag
