from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.exchange import timeframe_to_minutes
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Godie(IStrategy):


  minimal_roi = {
    "0": 0.1, # 5% for the first 3 candles
    "180": 0.05,  # 2% after 3 candles
    "360": 0.03,  # 1% After 6 candles
  }

  stoploss = -0.2

  timeframe = '15m'

  def wadda_macd(self, source, fast = 20, slow=40):
    fastMA = ta.EMA(source, timeperiod=fast)
    slowMA = ta.EMA(source, timeperiod=slow)
    return fastMA - slowMA
  
  def wadda_BBUpper(self, source, channelLength = 20, multiplier=2.0):
    basis = ta.SMA(source, timeperiod=channelLength)
    dev = multiplier * source.rolling(channelLength).std()
    return basis + dev
  
  def wadda_BBLower(self, source, channelLength = 20, multiplier=2.0):
    basis = ta.SMA(source, timeperiod=channelLength)
    dev = multiplier * source.rolling(channelLength).std()
    return basis - dev

  def vwap_bands(self, dataframe, window_size=20, num_of_std=1):
   df = dataframe.copy()
   df["vwap"] = qtpylib.rolling_vwap(df, window=window_size)
   rolling_std = df["vwap"].rolling(window=window_size).std()
   df["vwap_low"] = df["vwap"] - (rolling_std * num_of_std)
   df["vwap_high"] = df["vwap"] + (rolling_std * num_of_std)
   return df["vwap_low"], df["vwap"], df["vwap_high"]
  
  def count_cumulative_occurrences(dataframe):
    df = dataframe.copy()

    bajos = ~df['bajo']
    altos = ~df['alto']
    suma_acumulada_bajos = bajos.cumsum()
    suma_acumulada_altos = altos.cumsum()

    suma_acumulada_filtrada_altos = suma_acumulada_altos-suma_acumulada_altos.where(~altos).ffill().fillna(1).astype(int)
    suma_acumulada_filtrada_bajos = suma_acumulada_altos-suma_acumulada_bajos.where(~bajos).ffill().fillna(1).astype(int)

    df['last_high'] = suma_acumulada_filtrada_altos
    df['last_low'] = suma_acumulada_filtrada_bajos
    return df['last_high'], df['last_low']

  def version(self) -> str:
    """
    Returns version of the strategy.
    """
    return "0.1"

  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe['ema20_close'] = ta.EMA(dataframe['close'], timeperiod=20)
    dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
    dataframe['ema15'] = ta.EMA(dataframe, timeperiod=15)
    dataframe['rsi'] = ta.RSI(dataframe)

    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']

    vwap_low, vwap, vwap_high = self.vwap_bands(dataframe, 20, 1)
    dataframe['vwap'] = vwap
    dataframe['vwap_low'] = vwap_low
    dataframe['vwap_high'] = vwap_high

    period = 12
    exit_period = 10
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    sup = dataframe["high"].rolling(center=False, window=exit_period).max()
    sdown = dataframe["low"].rolling(center=False, window=exit_period).min()
    dataframe['highest_high'] = highest_high
    dataframe['lowest_low'] = lowest_low
    dataframe['sup'] = sup
    dataframe['sdown'] = sdown
    dataframe['alto'] = dataframe['high'] >= highest_high
    dataframe['bajo'] = dataframe['low'] <= lowest_low


    sensitivity = 150
    dataframe['trend_1'] = abs((self.wadda_macd(dataframe['close']) - self.wadda_macd(dataframe['close'].shift(1))) * sensitivity)
    dataframe['explosion'] = self.wadda_BBUpper(dataframe['close']) - self.wadda_BBLower(dataframe['close'])

 

    return dataframe

  def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['close'] < dataframe['vwap_low']) &
            (dataframe['high'] >= dataframe['highest_high'] ) &
            (dataframe['trend_1'] > dataframe['explosion']) &
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        'buy'] = 1


    return dataframe

  
  def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

    dataframe.loc[
        ( 
            (dataframe['close'] > dataframe['vwap_high']) &
            (dataframe['sdown'] >= dataframe['low'] ) &
            (dataframe['trend_1'] > dataframe['explosion']) &
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        'sell'] = 1
    return dataframe
  

 
