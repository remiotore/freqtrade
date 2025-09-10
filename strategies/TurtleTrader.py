from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta


class TurtlePyramidingStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Strategy Parameters
    entry_period = 20
    exit_period = 10
    atr_period = 20
    atr_mult = 2.0  # For stop loss
    pyr_atr_mult = 0.5  # For pyramiding levels

    # Freqtrade Settings
    timeframe = '1d'
    startup_candle_count: int = entry_period + atr_period
    minimal_roi = {"0": 0.10}
    stoploss = -0.99  # Use custom stoploss
    trailing_stop = False

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['20d_high'] = df['high'].rolling(window=self.entry_period).max()
        df['10d_low'] = df['low'].rolling(window=self.exit_period).min()
        df['atr'] = ta.ATR(df, timeperiod=self.atr_period)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = 0
        df.loc[
            (df['close'] > df['20d_high'].shift(1)),
            'enter_long'
        ] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = 0
        df.loc[
            (df['close'] < df['10d_low'].shift(1)),
            'exit_long'
        ] = 1
        return df

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, exit_tag: str, **kwargs) -> bool:
        trade = kwargs.get("trade", None)

        if not trade:
            return True  # Initial entry

        # Pyramiding logic
        data = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
        candle = data.iloc[-1]
        base_price = trade.open_rate
        atr = candle['atr']
        position_size = trade.amount

        # Determine pyramid level based on price distance
        levels_crossed = int((candle['close'] - base_price) / (self.pyr_atr_mult * atr))
        current_units = int(position_size / trade.open_trade_data.get('base_unit_size', 1.0))

        # Max 4 units total
        if current_units < 4 and levels_crossed > current_units:
            trade.open_trade_data['base_unit_size'] = position_size if 'base_unit_size' not in trade.open_trade_data else trade.open_trade_data['base_unit_size']
            return True
        return False

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        atr = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe).iloc[-1]['atr']
        stoploss_price = trade.open_rate - self.atr_mult * atr
        if current_rate <= stoploss_price:
            return 0.0
        return 1
