
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta



class SaulusStrategiaOtimizeHyperOpt(IStrategy):




    minimal_roi = {
        "0": 0.098,
        "29": 0.07,
        "84": 0.022,
        "146": 0
    }

    stoploss = -0.278

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    timeframe = '5m'

    startup_candle_count: int = 20

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
       dataframe['rsi'] = ta.RSI(dataframe)#mede a aceleração do movimento dos preços do ativo
       dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)#media movel simples 
       dataframe['sma80'] = ta.SMA(dataframe, timeperiod=80)#media movel simples  
       dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)#média exponencial movel
       
       return dataframe

     

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] < 30) & #uma ordem de compra é enviada se o ROI for menor que 30.
                    (dataframe['sma20'] >= dataframe['sma80']) < dataframe['ema5']
                    
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 50) & #uma ordem de venda é emitida quando o ROI for maior que 70
                    (dataframe['sma20'] <= dataframe['sma80']) > dataframe['ema5']
            ),
            'sell'] = 1
        return dataframe