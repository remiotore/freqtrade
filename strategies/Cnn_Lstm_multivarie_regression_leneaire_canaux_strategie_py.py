from freqtrade.strategy.interface import IStrategy
import numpy as np
from sklearn.linear_model import LinearRegression

class Cnn_Lstm_multivarie_regression_leneaire_canaux_strategie_py(IStrategy):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(dataframe[['feature1', 'feature2', 'feature3']])

        dataframe['prediction'] = predictions

        dataframe['upper_channel'] = predictions + 1.96 * model.scale_
        dataframe['lower_channel'] = predictions - 1.96 * model.scale_
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[dataframe['close'] < dataframe['lower_channel'], 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[dataframe['close'] > dataframe['upper_channel'], 'sell'] = 1

        return dataframe
