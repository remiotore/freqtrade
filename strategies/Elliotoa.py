from freqtrade.strategy import IStrategy
from freqtrade.exchange import timeframe_to_minutes
from pandas import DataFrame

class Elliotoa(IStrategy):
    minimal_roi = {
        "0": 0.02
    }

    stoploss = -0.02

    ema_period = 13
    risk_reward_ratio = 2.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema'] = dataframe['close'].ewm(span=self.ema_period, adjust=False).mean()

        dataframe['price_vs_ema'] = dataframe['close'] - dataframe['ema']

        dataframe['wave_pattern'] = 0

        dataframe.loc[dataframe['price_vs_ema'] > 0, 'wave_pattern'] = 1

        dataframe.loc[dataframe['price_vs_ema'] < 0, 'wave_pattern'] = -1

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['wave_pattern'].shift(1) == -1) & (dataframe['wave_pattern'] == 1),
            'buy_signal'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['wave_pattern'].shift(1) == 1) & (dataframe['wave_pattern'] == -1),
            'sell_signal'
        ] = 1

        return dataframe

    def populate_stop_loss(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['stop_loss'] = dataframe['close'] * self.risk_reward_ratio * self.stoploss

        return dataframe

    def populate_take_profit(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['take_profit'] = dataframe['close'] * self.risk_reward_ratio

        return dataframe
