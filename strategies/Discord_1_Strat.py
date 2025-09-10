class Strat(IStrategy):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        should_buy = [False] * len(dataframe)
        for i in range(0, len(dataframe)):
            dataframe_slice = dataframe[0:i + 1]
            macd_condition = (dataframe_slice['macd'] < dataframe_slice['macdsignal'])[i]
            should_buy[i] = macd_condition

        dataframe["should_buy"] = should_buy
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['should_buy'])
            ),
            'buy'] = 1
        return dataframe