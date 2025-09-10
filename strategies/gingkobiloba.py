from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta

class ImprovedTVStrategy(IStrategy):
    timeframe = '5m'
    
    # Пользовательские комиссии для расчета прибыли
    custom_fee_open_rate = None
    custom_fee_close_rate = None
    
    # Остальные параметры стратегии
    ema_length = IntParameter(10, 14, default=12, space='buy')
    sma_length = IntParameter(3, 7, default=5, space='buy')
    adx_length = IntParameter(12, 16, default=14, space='buy')
    adx_threshold = IntParameter(25, 35, default=30, space='buy')
    macd_short = IntParameter(10, 14, default=12, space='buy')
    macd_long = IntParameter(24, 28, default=26, space='buy')
    macd_signal = IntParameter(8, 10, default=9, space='buy')
    vwap_length = IntParameter(12, 16, default=14, space='buy')
    vwap_multiplier = DecimalParameter(1.5, 2.5, default=2.0, space='buy')
    # Параметры тейк-профита
    take_profit_1 = DecimalParameter(0.008, 0.012, default=0.01, space='sell')
    take_profit_2 = DecimalParameter(0.018, 0.022, default=0.02, space='sell')
    breakeven_offset = DecimalParameter(0.0008, 0.0012, default=0.001, space='sell')

    # Параметры риск-менеджмента
    stoploss = -0.02 # 2% стоп-лосс
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_length.value)
        
        # SMA
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.sma_length.value)
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_length.value)
        
        # MACD
        macd = ta.MACD(dataframe, 
                       fastperiod=self.macd_short.value,
                       slowperiod=self.macd_long.value,
                       signalperiod=self.macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Настоящий VWAP
        dataframe['vwap'] = (dataframe['volume'] * dataframe['close']).cumsum() / dataframe['volume'].cumsum()

        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
              (
                  (dataframe['macd'] > dataframe['macdsignal']) &  # MACD выше сигнальной
                  (dataframe['adx'] > self.adx_threshold.value) &   # ADX выше порога
                  (dataframe['close'] > dataframe['ema']) &         # Цена выше EMA
                  (dataframe['close'] > dataframe['vwap'])          # Цена выше VWAP
              ),
              'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
              (
                  (dataframe['macd'] < dataframe['macdsignal']) &  # MACD ниже сигнальной
                  (dataframe['adx'] > self.adx_threshold.value) &   # ADX выше порога
                  (dataframe['close'] < dataframe['ema']) &         # Цена ниже EMA
                  (dataframe['close'] < dataframe['vwap'])          # Цена ниже VWAP
              ),
              'exit_long'] = 1
        return dataframe
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
               current_profit: float, **kwargs):
        # Реализация тейк-профитов из TradingView стратегии
        if current_profit >= self.take_profit_2.value:
            return 'take_profit_2'
        elif current_profit >= self.take_profit_1.value:
            return 'take_profit_1'
        
        return None