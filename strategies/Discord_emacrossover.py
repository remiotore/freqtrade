class emacrossover(IStrategy):

    INTERFACE_VERSION = 2

    # 18233 trades. 8642/5823/3768 Wins/Draws/Losses. Avg profit   0.72%. Median profit   0.00%. Total profit  132158.95851787 USDT ( 13202.69?%). Avg duration 732.0 min. Objective: -11.51195

    # Hyperopt parameters
    buy_fastema = IntParameter(low=1, high=100, default=28, space='buy', optimize=True, load=True)
    buy_slowema = IntParameter(low=1, high=365, default=347, space='buy', optimize=True, load=True)
    sell_fastema = IntParameter(low=1, high=100, default=40, space='sell', optimize=True, load=True)
    sell_slowema = IntParameter(low=1, high=365, default=109, space='sell', optimize=True, load=True)

    # ROI table:
    minimal_roi = {
        "0": 0.16412,
        "54": 0.08957,
        "195": 0.04799,
        "555": 0
    }

    # Stoploss:
    stoploss = -0.32523


    timeframe = '15m'

    # def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     dataframe['buy-fastMA'] = ta.EMA(dataframe, timeperiod=self.buy_fastema.value)
    #     dataframe['buy-slowMA'] = ta.EMA(dataframe, timeperiod=self.buy_slowema.value)
    #     dataframe['sell-fastMA'] = ta.EMA(dataframe, timeperiod=self.sell_fastema.value)
    #     dataframe['sell-slowMA'] = ta.EMA(dataframe, timeperiod=self.sell_slowema.value)
    #     return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Trigger
        conditions.append(ta.EMA(dataframe, timeperiod=int(self.buy_fastema.value)) > ta.EMA(dataframe, timeperiod=int(self.buy_slowema.value)))
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Trigger
        conditions.append(ta.EMA(dataframe, timeperiod=int(self.sell_fastema.value)) < ta.EMA(dataframe, timeperiod=int(self.sell_slowema.value)))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe