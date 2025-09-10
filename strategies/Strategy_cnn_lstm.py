8from freqtrade.strategy.interface import IStrategy
import tensorflow as tf
import numpy as np

class Strategy_cnn_lstm(IStrategy):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        predictions = model.predict(dataframe)

        dataframe['prediction'] = predictions
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[dataframe['prediction'] > 0.5, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[dataframe['prediction'] < 0.5, 'sell'] = 1

        return dataframe
