import numpy as np
import pandas as pd
import pandas_ta as ta
import talib.abstract as talib
from technical import qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime


class crossAboveStrategy(IStrategy):

    timeframe = '5m'

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.2

    use_custom_stoploss = True

    exit_profit_only = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:

        # Retrieves the analyzed dataframe for the specified currency pair and timeframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # Extracts the last candle (most recent data point) from the dataframe
        last_candle = dataframe.iloc[-1].squeeze()

        if trade.stop_loss == trade.initial_stop_loss:
            # If current stoploss is same as the initial stoploss value,
            # update it based on ATR for last candle
            atr_stoploss = stoploss_from_absolute(last_candle['ATR_stoploss'], current_rate)
            if np.isnan(atr_stoploss):
                return None
            else:
                return atr_stoploss

        # Return current stoploss if none of the above conditions are met
        return trade.stop_loss

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        # Convert 'date' column to datetime format and store it in a new column
        dataframe['date_copy'] = pd.to_datetime(dataframe['date'])
        # Remove timezone information from 'date_copy' column
        dataframe['date_copy'] = dataframe['date_copy'].dt.tz_localize(None)
        # Set 'date_copy' column as the DataFrame index
        dataframe.set_index('date_copy', inplace=True)

        # Calclate ATR
        dataframe['ATR'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=150)
        # Calculate what will be used as a ATR-based stoploss
        dataframe['ATR_stoploss'] = dataframe['close'] - dataframe['ATR'] * 6.5

        # Calculate and add the Relative Strength Index (RSI) to the dataframe
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=16)

        # Calculate ema 50
        dataframe['ema50'] = talib.EMA(dataframe, timeperiod = 50)
        dataframe['ema100'] = talib.EMA(dataframe, timeperiod = 100)
        dataframe['ema200'] = talib.EMA(dataframe, timeperiod = 200)

        # Calculate candle sum
        candle_sum = dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']
        # Calculate mean value per candle
        dataframe['candle_mean'] = candle_sum / 4

        # Calculate the 14-period Simple Moving Average (SMA) - (Average closing price for 14 candles)
        sma = dataframe['close'].rolling(20).mean()
        # Calculate the Standard Deviation for the same period
        std_dev = dataframe['close'].rolling(20).std()

        # Calculate the upper and lower Bollinger Band
        dataframe['upper_band'] = sma + (std_dev * 2.0)
        dataframe['lower_band'] = sma - (std_dev * 2.0)

        # If candle_mean is above or equal ema100, set ema_signal to 1
        ema_signal = 0
        dataframe['ema_signal'] = ema_signal
        dataframe.loc[(dataframe['candle_mean'] >= dataframe['ema100'])
                      , 'ema_signal'] = 1
        
        # If candle_mean has crossed above ema100 the last 3 candles, set up_trend to 1
        up_trend = 0
        dataframe['up_trend'] = up_trend
        crossed_above = qtpylib.crossed_above(dataframe['candle_mean'], dataframe['ema100'])
        last_three = crossed_above.rolling(3).sum() > 0
        dataframe.loc[last_three, 'up_trend'] = 1

        # Calculate the percentile over rolling windows of data
        dataframe['rsi_pnr_high'] = percentile_nearest_rank(dataframe['rsi'], 300, 80)

        # Calculate the percentile over rolling windows of data
        dataframe['rsi_pnr_low'] = percentile_nearest_rank(dataframe['rsi'], 300, 50)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                # Buy when volume > 0
                (dataframe['volume'] > 0) &
                # Buy when ema_signal is 1 - indicating bullish trend)
                (dataframe['ema_signal'] == 1) &
                # # Buy when up_trend is 1 - candle_mean has crossed above ema100 the past 3 dandles
                (dataframe['up_trend'] == 1) &
                # Make sure lower and upper bband is not the same (no momentum)
                (dataframe['lower_band'] != dataframe['upper_band'])
            ),
            # If all conditions are True for a given row, the 'buy' column for that row is set to 1,
            'buy'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (
                # Sell when volume > 0
                (dataframe['volume'] > 0) &
                # Sell when closing price is the same or above the upper bband
                (dataframe['close'] >= dataframe['upper_band']) &
                # Sell when rsi is historically high
                (dataframe['rsi'] > dataframe['rsi_pnr_high']) &
                # Make sure lower and upper bband is not the same (no momentum)
                (dataframe['lower_band'] != dataframe['upper_band'])
            ),
            # If all conditions are True for a given row, the 'sell' column for that row is set to 1
            'sell'
        ] = 1

        return dataframe
    
def percentile_nearest_rank(series, length, percentile):
    def rolling_percentile(x, p):
        return np.percentile(x, p)

    return series.rolling(window=length).apply(rolling_percentile, args=(percentile,), raw=True)
