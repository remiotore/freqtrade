from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib as ta


class Smart5MinStrategy(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.04,
        "30": 0.02,
        "60": 0
    }

    # Static stoploss (fallback if trailing not hit)
    stoploss = -0.10

    # Custom trailing stop settings
    trailing_stop = True
    trailing_stop_positive = 0.01           # 1% profit to start trailing
    trailing_stop_positive_offset = 0.02    # 2% profit to set the stoploss
    trailing_only = True                    # only use trailing stop, ignore static stoploss after offset

    # Use volume in the strategy
    use_volume = True

    # Indicator parameters
    ema_period = 50
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    drop_pct = 0.02          # Threshold for a "big drop" (2%)
    drop_period = 1          # Period over which to measure drop (bars)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several indicators needed for entry and exit points
        """
        # EMA
        dataframe['ema50'] = ta.EMA(dataframe['close'], timeperiod=self.ema_period)

        # MACD
        macd_line, signal_line, _ = ta.MACD(
            dataframe['close'],
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        dataframe['macd'] = macd_line
        dataframe['macdsignal'] = signal_line

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period)

        # Volume average for confirmation
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        # Price drop percentage over last bar(s)
        dataframe['close_shift'] = dataframe['close'].shift(self.drop_period)
        dataframe['drop_pct'] = (dataframe['close_shift'] - dataframe['close']) / dataframe['close_shift']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the 'enter_long' column with 1 when entry conditions are met
        Conditions:
        - Big drop (> drop_pct)
        - RSI oversold (<30)
        - Volume spike (> 20-bar mean)
        """
        # Entry signals
        cond_drop = dataframe['drop_pct'] > self.drop_pct
        cond_rsi = dataframe['rsi'] < 30
        cond_vol = dataframe['volume'] > dataframe['volume_mean']

        dataframe.loc[
            cond_drop & cond_rsi & cond_vol,
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the 'exit_long' column with 1 when exit conditions are met
        Conditions:
        - MACD bearish crossover
        - RSI overbought (>70)
        - Price below EMA50
        (Trailing stop will also manage exits based on profit)
        """
        cond_exit_macd = dataframe['macd'] < dataframe['macdsignal']
        cond_exit_rsi = dataframe['rsi'] > 70
        cond_price_below_ema = dataframe['close'] < dataframe['ema50']

        dataframe.loc[
            (cond_exit_macd | cond_exit_rsi | cond_price_below_ema),
            'exit_long'
        ] = 1

        return dataframe
