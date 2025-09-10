

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.a2c.a2c import A2C


class FreqGym_normalized_412(IStrategy):









    minimal_roi = {
        "0": 100
    }

    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.017
    trailing_only_offset_is_reached = True


    ticker_interval = '5m'

    use_sell_signal = True

    process_only_new_candles = False

    startup_candle_count: int = 200

    model = None
    window_size = None

    try:
        model = PPO.load('models/best_model')  # Note: Make sure you use the same policy as the one used to train
        window_size = model.observation_space.shape[0]
    except Exception:
        pass

    timeperiods = [7, 14, 21]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        dataframe['plus_di'] = normalize(ta.PLUS_DI(dataframe), 0, 100)

        dataframe['minus_di'] = normalize(ta.MINUS_DI(dataframe), 0, 100)

        dataframe['uo'] = normalize(ta.ULTOSC(dataframe), 0, 100)

        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = normalize(hilbert['sine'], -1, 1)
        dataframe['htleadsine'] = normalize(hilbert['leadsine'], -1, 1)

        dataframe['bop'] = normalize(ta.BOP(dataframe), -1, 1)

        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = normalize(stoch['slowk'], 0, 100)
        dataframe['slowd'] = normalize(stoch['slowd'], 0, 100)

        stochf = ta.STOCHF(dataframe)
        dataframe['fastk'] = normalize(stochf['fastk'], 0, 100)
        dataframe['fastk'] = normalize(stochf['fastk'], 0, 100)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)

        dataframe['bb2_lower_gt_close'] = bollinger2['lower'].gt(dataframe['close']).astype('int')
        dataframe['bb3_lower_gt_close'] = bollinger3['lower'].gt(dataframe['close']).astype('int')

        for period in self.timeperiods:

            dataframe[f'adx_{period}'] = normalize(ta.ADX(dataframe, timeperiod=period), 0, 100)

            aroon = ta.AROON(dataframe, timeperiod=period)
            dataframe[f'aroonup_{period}'] = normalize(aroon['aroonup'], 0, 100)
            dataframe[f'aroondown_{period}'] = normalize(aroon['aroondown'], 0, 100)
            dataframe[f'aroonosc_{period}'] = normalize(ta.AROONOSC(dataframe, timeperiod=period), -100, 100)

            dataframe[f'cmo_{period}'] = normalize(ta.CMO(dataframe, timeperiod=period), -100, 100)

            dataframe[f'dx_{period}'] = normalize(ta.DX(dataframe, timeperiod=period), 0, 100)

            dataframe[f'mfi_{period}'] = normalize(ta.MFI(dataframe, timeperiod=period), 0, 100)

            dataframe[f'minus_di_{period}'] = normalize(ta.MINUS_DI(dataframe, timeperiod=period), 0, 100)

            dataframe[f'plus_di_{period}'] = normalize(ta.PLUS_DI(dataframe, timeperiod=period), 0, 100)

            dataframe[f'willr_{period}'] = normalize(ta.WILLR(dataframe, timeperiod=period), -100, 0)

            dataframe[f'rsi_{period}'] = normalize(ta.RSI(dataframe, timeperiod=period), 0, 100)

            rsi = 0.1 * (dataframe[f'rsi_{period}'] - 50)
            dataframe[f'fisher_rsi_{period}'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
            dataframe[f'fisher_rsi_{period}'] = normalize(dataframe[f'fisher_rsi_{period}'], -1, 1)

            stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=period)
            dataframe[f'stochrsi_k_{period}'] = normalize(stoch_rsi['fastk'], 0, 100)
            dataframe[f'stochrsi_d_{period}'] = normalize(stoch_rsi['fastd'], 0, 100)



            dataframe[f'linangle_{period}'] = normalize(ta.LINEARREG_ANGLE(dataframe, timeperiod=period), -90, 90)


        indicators = dataframe[dataframe.columns[~dataframe.columns.isin(['date', 'open', 'close', 'high', 'low', 'volume', 'buy', 'sell', 'buy_tag'])]]

        assert all(indicators.max() < 1.00001) and all(indicators.min() > -0.00001), "Error, values are not normalized!"

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        action = self.rl_model_predict(dataframe)
        dataframe['buy'] = (action == 1).astype('int')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        action = self.rl_model_predict(dataframe)
        dataframe['sell'] = (action == 2).astype('int')

        return dataframe

    def rl_model_predict(self, dataframe):
        output = pd.DataFrame(np.zeros((len(dataframe), 1)))
        indicators = dataframe[dataframe.columns[~dataframe.columns.isin(['date', 'open', 'close', 'high', 'low', 'volume', 'buy', 'sell', 'buy_tag'])]].fillna(0).to_numpy()

        for window in range(self.window_size, len(dataframe)):
            start = window - self.window_size
            end = window
            observation = indicators[start:end]
            res, _ = self.model.predict(observation, deterministic=True)
            output.loc[end] = res

        return output

def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)