from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class AdvancedMultiIndicatorStrategy(IStrategy):

    minimal_roi = {
        "0": 0.15,
        "10": 0.1,
        "30": 0
    }

    stoploss = -0.08
    timeframe = '5m'
    startup_candle_count = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['signal']

        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 35) &  # RSI 超卖
                (dataframe['macd'] > dataframe['macd_signal']) &  # MACD 金叉
                (dataframe['stoch_k'] < 20) &  # Stochastic 超卖
                (dataframe['close'] < dataframe['bb_lower']) &  # 价格低于布林带下轨
                (dataframe['close'] > dataframe['ema_20']) &  # 价格高于 EMA 20
                (dataframe['adx'] > 25) &  # ADX 表示强趋势
                (dataframe['mfi'] < 20) &  # MFI 显示资金流入减少
                (dataframe['atr'] > 1.5 * dataframe['atr'].rolling(window=14).mean())  # 波动率增加
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 65) &  # RSI 超买
                (dataframe['macd'] < dataframe['macd_signal']) &  # MACD 死叉
                (dataframe['stoch_k'] > 80) &  # Stochastic 超买
                (dataframe['close'] > dataframe['bb_upper']) &  # 价格高于布林带上轨
                (dataframe['adx'] > 25) &  # ADX 表示强趋势
                (dataframe['mfi'] > 80) &  # MFI 显示资金流入增加
                (dataframe['atr'] < dataframe['atr'].rolling(window=14).mean())  # 波动率回落
            ),
            'sell'] = 1
        return dataframe

