from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class XMRMultiTimeframeStrategy(IStrategy):
    timeframe = '5m'
    inf_timeframe = '15m'
    can_short = True  # 合约交易支持做空
    leverage = 3

    startup_candle_count: int = 50

    # ROI、止损、Trailing Stop
    minimal_roi = {"0": 0.04}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    process_only_new_candles = True
    use_custom_stoploss = False
    ignore_buy_signal_if_still_in_trade = True

    def informative_pairs(self):
        # 注册 15m 数据用作大周期分析
        pairs = [(pair, self.inf_timeframe) for pair in self.dp.current_whitelist]
        return pairs

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # 当前 5m 图表指标
        df['rsi'] = ta.RSI(df, timeperiod=14)
        bb = ta.BBANDS(df['close'], timeperiod=20)
        df['bb_upperband'] = bb['upperband']
        df['bb_middleband'] = bb['middleband']
        df['bb_lowerband'] = bb['lowerband']

        df['ema_fast'] = ta.EMA(df['close'], timeperiod=9)
        df['ema_slow'] = ta.EMA(df['close'], timeperiod=21)

        return df

    def populate_indicators_inf(self, df: DataFrame, metadata: dict) -> DataFrame:
        # 15m 图表指标（趋势过滤器）
        df['ema_fast_15'] = ta.EMA(df['close'], timeperiod=9)
        df['ema_slow_15'] = ta.EMA(df['close'], timeperiod=21)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = 0
        df['enter_short'] = 0

        inf_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        df = df.merge(
            inf_df[['ema_fast_15', 'ema_slow_15']],
            left_index=True, right_index=True, how='left'
        )

        # 多头：15m EMA 向上 + 5m RSI < 35 + 收盘低于中轨（低吸）
        df.loc[
            (
                df['ema_fast_15'] > df['ema_slow_15']
                & (df['rsi'] < 35)
                & (df['close'] < df['bb_middleband'])
                & (df['close'] > df['ema_fast'])
            ),
            'enter_long'
        ] = 1

        # 空头：15m EMA 向下 + 5m RSI > 65 + 收盘高于中轨（高抛）
        df.loc[
            (
                df['ema_fast_15'] < df['ema_slow_15']
                & (df['rsi'] > 65)
                & (df['close'] > df['bb_middleband'])
                & (df['close'] < df['ema_fast'])
            ),
            'enter_short'
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = 0
        df['exit_short'] = 0

        # 多头退出：超买或接近上轨
        df.loc[
            (df['rsi'] > 70) | (df['close'] > df['bb_upperband']),
            'exit_long'
        ] = 1

        # 空头退出：超卖或接近下轨
        df.loc[
            (df['rsi'] < 30) | (df['close'] < df['bb_lowerband']),
            'exit_short'
        ] = 1

        return df
