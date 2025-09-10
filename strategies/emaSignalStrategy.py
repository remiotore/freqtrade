import numpy as np
import pandas as pd
import pandas_ta as ta
import talib.abstract as talib
from technical import qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime

class emaSignalStrategy(IStrategy):

    INTERFACE_VERSION = 2

    timeframe = '15m'

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.2

    use_custom_stoploss = True

    exit_profit_only = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if trade.stop_loss == trade.initial_stop_loss:


            atr_stoploss = stoploss_from_absolute(last_candle['ATR_stoploss'], current_rate)
            if np.isnan(atr_stoploss):
                return None
            else:
                return atr_stoploss

        return trade.stop_loss

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['date_copy'] = pd.to_datetime(dataframe['date'])

        dataframe['date_copy'] = dataframe['date_copy'].dt.tz_localize(None)

        dataframe.set_index('date_copy', inplace=True)

        dataframe['ATR'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=150)

        dataframe['ATR_stoploss'] = dataframe['close'] - dataframe['ATR'] * 6.5

        dataframe['rsi'] = ta.rsi(dataframe['close'], length=16)

        dataframe['ema50'] = talib.EMA(dataframe, timeperiod = 50)
        dataframe['ema100'] = talib.EMA(dataframe, timeperiod = 100)
        dataframe['ema200'] = talib.EMA(dataframe, timeperiod = 200)

        candle_sum = dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']

        dataframe['candle_mean'] = candle_sum / 4

        sma = dataframe['close'].rolling(20).mean()

        std_dev = dataframe['close'].rolling(20).std()

        dataframe['upper_band'] = sma + (std_dev * 2.0)
        dataframe['lower_band'] = sma - (std_dev * 2.0)

        ema_signal = 0
        dataframe['ema_signal'] = ema_signal
        dataframe.loc[(dataframe['candle_mean'] >= dataframe['ema100'])
                      , 'ema_signal'] = 1

        dataframe['rsi_pnr_high'] = percentile_nearest_rank(dataframe['rsi'], 300, 80)

        dataframe['rsi_pnr_low'] = percentile_nearest_rank(dataframe['rsi'], 300, 50)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (

                (dataframe['volume'] > 0) &

                (dataframe['ema_signal'] == 1) &

                (dataframe['rsi'] < dataframe['rsi_pnr_low']) &

                (dataframe['close'] <= dataframe['lower_band']) &

                (dataframe['close'].shift(1) <= dataframe['lower_band'].shift(1)) &

                (dataframe['lower_band'] != dataframe['upper_band'])
            ),

            'buy'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (

                (dataframe['volume'] > 0) &

                (dataframe['close'] >= dataframe['upper_band']) &

                (dataframe['rsi'] > dataframe['rsi_pnr_high']) &

                (dataframe['lower_band'] != dataframe['upper_band'])
            ),

            'sell'
        ] = 1

        return dataframe
    
def percentile_nearest_rank(series, length, percentile):
    def rolling_percentile(x, p):
        return np.percentile(x, p)

    return series.rolling(window=length).apply(rolling_percentile, args=(percentile,), raw=True)
