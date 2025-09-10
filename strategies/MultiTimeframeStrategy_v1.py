import math
import numpy as np
import pandas as pd
import talib as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union, Tuple
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter, informative

class MultiTimeframeStrategy_v1(IStrategy):
  INTERFACE_VERSION = 3
  can_short: bool = True
  levarage_input = 10.0
  stoploss = -0.058
  trailing_stop = False
  trailing_only_offset_is_reached = True
  trailing_stop_positive = 0.019
  trailing_stop_positive_offset = 0.053
  position_adjustment_enable = True
  timeframe = "5m"
  minimal_roi = {
      "0": 0.113,
      "29": 0.074,
      "54": 0.037,
      "76": 0,
  }
  process_only_new_candles = True
  use_exit_signal = True
  exit_profit_only = True
  ignore_roi_if_entry_signal = False

  buy_params = {
      "buy_rsi_5m": 20, "buy_rsi_1h": 12, "buy_rsi_4h": 19,
      "buy_rsi_compare_long_5m": 33, "buy_rsi_compare_long_1h": 52, "buy_rsi_compare_long_4h": 29,
      "buy_rsi_compare_short_5m": 54, "buy_rsi_compare_short_1h": 51, "buy_rsi_compare_short_4h": 63,
      "long_multiplier_5m": 0.985, "long_multiplier_1h": 0.961,
      "short_multiplier_5m": 1.01, "short_multiplier_1h": 1.026,
      "buy_factor_5m": 3.011, "buy_factor_1h": 3.55,
      "buy_period_5m": 13, "buy_period_1h": 7,
  }
  sell_params = {
      "sell_rsi_compare_long_5m": 65, "sell_rsi_compare_long_1h": 71,
      "sell_rsi_compare_short_5m": 25, "sell_rsi_compare_short_1h": 33,
      "sell_decr_pos_5m": 0.115, "sell_decr_pos_1h": 0.265
  }

  buy_rsi_5m = IntParameter(8, 20, default=buy_params['buy_rsi_5m'], space="buy", optimize=True)
  buy_rsi_1h = IntParameter(8, 20, default=buy_params['buy_rsi_1h'], space="buy", optimize=True)
  buy_rsi_4h = IntParameter(8, 20, default=buy_params['buy_rsi_4h'], space="buy", optimize=True)

  buy_rsi_compare_long_5m = IntParameter(20, 65, default=buy_params['buy_rsi_compare_long_5m'], space="buy", optimize=True)
  buy_rsi_compare_long_1h = IntParameter(20, 65, default=buy_params['buy_rsi_compare_long_1h'], space="buy", optimize=True)
  buy_rsi_compare_long_4h = IntParameter(20, 65, default=buy_params['buy_rsi_compare_long_4h'], space="buy", optimize=True)

  buy_rsi_compare_short_5m = IntParameter(20, 65, default=buy_params['buy_rsi_compare_short_5m'], space="buy", optimize=True)
  buy_rsi_compare_short_1h = IntParameter(20, 65, default=buy_params['buy_rsi_compare_short_1h'], space="buy", optimize=True)
  buy_rsi_compare_short_4h = IntParameter(20, 65, default=buy_params['buy_rsi_compare_short_4h'], space="buy", optimize=True)

  sell_rsi_compare_long_5m = IntParameter(60, 80, default=sell_params['sell_rsi_compare_long_5m'], space="sell", optimize=True)
  sell_rsi_compare_long_1h = IntParameter(60, 80, default=sell_params['sell_rsi_compare_long_1h'], space="sell", optimize=True)

  sell_rsi_compare_short_5m = IntParameter(20, 40, default=sell_params['sell_rsi_compare_short_5m'], space="sell", optimize=True)
  sell_rsi_compare_short_1h = IntParameter(20, 40, default=sell_params['sell_rsi_compare_short_1h'], space="sell", optimize=True)

  long_multiplier_5m = DecimalParameter(0.95, 1.0, default=buy_params['long_multiplier_5m'], decimals=3, space='buy', optimize=True)
  long_multiplier_1h = DecimalParameter(0.95, 1.0, default=buy_params['long_multiplier_1h'], decimals=3, space='buy', optimize=True)

  short_multiplier_5m = DecimalParameter(1.0, 1.05, default=buy_params['short_multiplier_5m'], decimals=3, space='buy', optimize=True)
  short_multiplier_1h = DecimalParameter(1.0, 1.05, default=buy_params['short_multiplier_1h'], decimals=3, space='buy', optimize=True)

  buy_factor_5m = DecimalParameter(1, 5, default=buy_params['buy_factor_5m'], decimals=3, space="buy", optimize=True)
  buy_factor_1h = DecimalParameter(1, 5, default=buy_params['buy_factor_1h'], decimals=3, space="buy", optimize=True)

  buy_period_5m = IntParameter(4, 15, default=buy_params['buy_period_5m'], space="buy", optimize=True)
  buy_period_1h = IntParameter(4, 15, default=buy_params['buy_period_1h'], space="buy", optimize=True)

  sell_decr_pos_5m = DecimalParameter(0.01, 0.3, decimals=3, default=sell_params['sell_decr_pos_5m'], space="sell", optimize=True)
  sell_decr_pos_1h = DecimalParameter(0.01, 0.3, decimals=3, default=sell_params['sell_decr_pos_1h'], space="sell", optimize=True)

  @informative('1h')
  def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
      dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.buy_rsi_1h.value)  
      dataframe['ssl_up'], dataframe['ssl_down'] = self.SSLChannels(dataframe, 10)

      dataframe["adx"] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'])  
      dataframe['WMA1'] = ta.WMA(dataframe["close"], timeperiod=30)
      dataframe['WMA2'] = ta.WMA(dataframe["close"], timeperiod=60)
      dataframe['SSL'] = ta.WMA(dataframe['WMA1'] - dataframe['WMA2'], timeperiod=int(math.sqrt(60)))

      return dataframe

  @informative('4h')
  def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
      dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.buy_rsi_4h.value)  
      dataframe['ssl_up'], dataframe['ssl_down'] = self.SSLChannels(dataframe, 10)
      return dataframe

  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

      dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.buy_rsi_5m.value)  
      dataframe['ssl_up'], dataframe['ssl_down'] = self.SSLChannels(dataframe, 10)

      
      dataframe["adx"] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'])  
      dataframe['WMA1'] = ta.WMA(dataframe["close"], timeperiod=30)
      dataframe['WMA2'] = ta.WMA(dataframe["close"], timeperiod=60)
      dataframe['SSL'] = ta.WMA(dataframe['WMA1'] - dataframe['WMA2'], timeperiod=int(math.sqrt(60)))

      return dataframe

  def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
      conditions = {}

      conditions['long_1h'] = (
          (dataframe['rsi_1h'] < self.buy_rsi_compare_long_1h.value) &
          (dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) &
          (dataframe['SSL_1h'] > 0) &
          (dataframe['ssl_up_4h'] > dataframe['ssl_down_4h'])  
      )
      conditions['short_1h'] = (
          (dataframe['rsi_1h'] > self.buy_rsi_compare_short_1h.value) &
          (dataframe['ssl_up_1h'] < dataframe['ssl_down_1h']) &
          (dataframe['SSL_1h'] < 0) &
          (dataframe['ssl_up_4h'] < dataframe['ssl_down_4h'])  
      )

      conditions['long_5m'] = (
          (dataframe['rsi'] < self.buy_rsi_compare_long_5m.value) &
          (dataframe['ssl_up'] > dataframe['ssl_down']) &
          ((dataframe['ssl_up_1h'] > dataframe['ssl_down_1h']) | (dataframe['ssl_up_4h'] > dataframe['ssl_down_4h']))  
      )
      conditions['short_5m'] = (
          (dataframe['rsi'] > self.buy_rsi_compare_short_5m.value) &
          (dataframe['ssl_up'] < dataframe['ssl_down']) &
          ((dataframe['ssl_up_1h'] < dataframe['ssl_down_1h']) | (dataframe['ssl_up_4h'] < dataframe['ssl_down_4h']))  
      )

      dataframe.loc[conditions['long_1h'], 'enter_long'] = 1
      dataframe.loc[conditions['long_1h'], 'enter_tag'] = '1h_long'

      dataframe.loc[conditions['long_5m'], 'enter_long'] = 1
      dataframe.loc[conditions['long_5m'], 'enter_tag'] = '5m_long'

      dataframe.loc[conditions['short_1h'], 'enter_short'] = 1
      dataframe.loc[conditions['short_1h'], 'enter_tag'] = '1h_short'

      dataframe.loc[conditions['short_5m'], 'enter_short'] = 1
      dataframe.loc[conditions['short_5m'], 'enter_tag'] = '5m_short'

      return dataframe

  def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

      dataframe.loc[
          (dataframe['rsi'] > self.sell_rsi_compare_long_5m.value) &
          (dataframe['rsi_1h'] > self.sell_rsi_compare_long_1h.value) &
          (dataframe['ssl_up'] < dataframe['ssl_down']),
          'exit_long'] = 1

      dataframe.loc[
          (dataframe['rsi'] < self.sell_rsi_compare_short_5m.value) &
          (dataframe['rsi_1h'] < self.sell_rsi_compare_short_1h.value) &
          (dataframe['ssl_up'] > dataframe['ssl_down']),
          'exit_short'] = 1

      return dataframe

  def custom_entry_price(self, pair: str, trade: Optional["Trade"], current_time: datetime, proposed_rate: float,
                         entry_tag: Optional[str], side: str, **kwargs) -> float:
      dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
      last_candle = dataframe.iloc[-1].squeeze()

      if side == "long":
          if entry_tag == '1h_long':
              return last_candle['close'] * self.long_multiplier_1h.value
          elif entry_tag == '5m_long':
              return last_candle['close'] * self.long_multiplier_5m.value
      else:
          if entry_tag == '1h_short':
              return last_candle['close'] * self.short_multiplier_1h.value
          elif entry_tag == '5m_short':
              return last_candle['close'] * self.short_multiplier_5m.value

      return proposed_rate

  def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Union[Optional[float], Tuple[Optional[float], Optional[str]]]:
      if trade.enter_tag == '1h_long' or trade.enter_tag == '1h_short':
          if current_profit > self.sell_decr_pos_1h.value and trade.nr_of_successful_exits == 0:
              return -(trade.stake_amount / 2), 'half_profit_1h'
      elif trade.enter_tag == '5m_long' or trade.enter_tag == '5m_short':
          if current_profit > self.sell_decr_pos_5m.value and trade.nr_of_successful_exits == 0:
              return -(trade.stake_amount / 2), 'half_profit_5m'
      return None

  def leverage(self, pair: str, current_time: datetime, current_rate: float,
               proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
               **kwargs) -> float:
      if self.levarage_input > max_leverage:
          return max_leverage
      return self.levarage_input

  def SSLChannels(self, dataframe, length=7):
      df = dataframe.copy()
      df["ATR"] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
      df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
      df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
      df["hlv"] = np.where(
          df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
      )
      df["hlv"] = df["hlv"].ffill()
      df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
      df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
      return df["sslDown"], df["sslUp"]

  def supertrend(self, dataframe: DataFrame, multiplier, period):
      df = dataframe.copy()

      df['ATR'] = ta.ATR(df['high'], df['low'], df['close'], period)

      st = 'ST_' + str(period) + '_' + str(multiplier)
      stx = 'STX_' + str(period) + '_' + str(multiplier)
      df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
      df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
      df['final_ub'] = 0.00
      df['final_lb'] = 0.00
      for i in range(period, len(df)):
          df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
              df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else \
              df['final_ub'].iat[i - 1]
          df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
              df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else \
              df['final_lb'].iat[i - 1]
      df[st] = 0.00
      for i in range(period, len(df)):
          df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[
              i] <= df['final_ub'].iat[i] else \
              df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > \
              df['final_ub'].iat[i] else \
              df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= \
              df['final_lb'].iat[i] else \
              df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < \
              df['final_lb'].iat[i] else 0.00
      df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)
      df['final_band'] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), df['final_lb'], df['final_ub']),
                                  np.NaN)
      df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
      return DataFrame(index=df.index, data={
          'ST': df[st],
          'STX': df[stx],
          'Fin_band': df['final_band']
      })

  def SSLChannels_ATR(self, dataframe, length=7):
      df = dataframe.copy()
      df["ATR"] = ta.ATR(df, timeperiod=14)
      df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
      df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
      df["hlv"] = np.where(
          df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN)
      )
      df["hlv"] = df["hlv"].ffill()
      df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
      df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
      return df["sslDown"], df["sslUp"]
