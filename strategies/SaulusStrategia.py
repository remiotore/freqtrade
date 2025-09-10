
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta



class SaulusStrategia(IStrategy):



    minimal_roi = {
        "0": 0.01, #liquidação dos ativos a qualquer momento independente da estratégia se um percentual de lucro de 70% for alcançado.
        "30": 0.005 #se a negociação estiver aberta por mais de 30 minutos, uma venda será emitida se os lucros excederem 20%
    }

    stoploss = -0.10

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