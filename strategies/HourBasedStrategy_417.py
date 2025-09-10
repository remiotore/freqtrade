








from freqtrade.strategy import IntParameter, IStrategy
from pandas import DataFrame







class HourBasedStrategy_417(IStrategy):



























    buy_params = {
        "buy_hour_max": 24,
        "buy_hour_min": 4,
    }

    sell_params = {
        "sell_hour_max": 21,
        "sell_hour_min": 22,
    }

    minimal_roi = {
        "0": 0.528,
        "169": 0.113,
        "528": 0.089,
        "1837": 0
    }

    stoploss = -0.10

    timeframe = '1h'

    buy_hour_min = IntParameter(0, 24, default=1, space='buy')
    buy_hour_max = IntParameter(0, 24, default=0, space='buy')

    sell_hour_min = IntParameter(0, 24, default=1, space='sell')
    sell_hour_max = IntParameter(0, 24, default=0, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['hour'] = dataframe['date'].dt.hour
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        min, max = self.buy_hour_min.value, self.buy_hour_max.value
        dataframe.loc[
            (
                (dataframe['hour'].between(min, max))
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        min, max = self.sell_hour_min.value, self.sell_hour_max.value
        dataframe.loc[
            (
                (dataframe['hour'].between(max, min))
            ),
            'sell'] = 1
        return dataframe
