
import logging
import time
from pathlib import Path

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib as ta
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from stable_baselines3 import A2C
from stable_baselines3.ppo.ppo import PPO

import predict

logger = logging.getLogger(__name__)

COLUMNS_FILTER = [
    'date',
    'open',
    'close',
    'high',
    'low',
    'volume',
    'buy',
    'sell',
    'buy_tag',
    'exit_tag',
]


class FreqGym_normalized_2(IStrategy):









    minimal_roi = {"0": 100}

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

    timeperiods = [7, 14, 21]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        try:





            model_file = Path(
                'models/best_model_FreqGymNormalized_FreqtradeEnv_A2C_20220318_115558.zip'
            )
            assert model_file.exists(), f'Model file "{model_file}" does not exist.'
            self.model = A2C.load(
                str(model_file)
            )  # Note: Make sure you use the same policy as the one used to train
            self.window_size = self.model.observation_space.shape[0]
        except Exception as e:
            logger.exception(f'Could not load model: {e}')
        else:
            logger.info(f'Loaded model: {model_file}')

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
        logger.info(f'Calculating TA indicators for {metadata["pair"]}')

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
            dataframe[f'aroonosc_{period}'] = normalize(
                ta.AROONOSC(dataframe, timeperiod=period), -100, 100
            )

            dataframe[f'cmo_{period}'] = normalize(ta.CMO(dataframe, timeperiod=period), -100, 100)

            dataframe[f'dx_{period}'] = normalize(ta.DX(dataframe, timeperiod=period), 0, 100)

            dataframe[f'mfi_{period}'] = normalize(ta.MFI(dataframe, timeperiod=period), 0, 100)

            dataframe[f'minus_di_{period}'] = normalize(
                ta.MINUS_DI(dataframe, timeperiod=period), 0, 100
            )

            dataframe[f'plus_di_{period}'] = normalize(
                ta.PLUS_DI(dataframe, timeperiod=period), 0, 100
            )

            dataframe[f'willr_{period}'] = normalize(
                ta.WILLR(dataframe, timeperiod=period), -100, 0
            )

            dataframe[f'rsi_{period}'] = normalize(ta.RSI(dataframe, timeperiod=period), 0, 100)

            rsi = 0.1 * (dataframe[f'rsi_{period}'] - 50)
            dataframe[f'fisher_rsi_{period}'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
            dataframe[f'fisher_rsi_{period}'] = normalize(dataframe[f'fisher_rsi_{period}'], -1, 1)

            stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=period)
            dataframe[f'stochrsi_k_{period}'] = normalize(stoch_rsi['fastk'], 0, 100)
            dataframe[f'stochrsi_d_{period}'] = normalize(stoch_rsi['fastd'], 0, 100)



            dataframe[f'linangle_{period}'] = normalize(
                ta.LINEARREG_ANGLE(dataframe, timeperiod=period), -90, 90
            )

        indicators = dataframe[dataframe.columns[~dataframe.columns.isin(COLUMNS_FILTER)]]

        assert all(indicators.max() < 1.00001) and all(
            indicators.min() > -0.00001
        ), "Error, values are not normalized!"
        logger.info(f'{metadata["pair"]} - indicators populated!')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        logger.info(f'Populating buy signal for {metadata["pair"]}')
        action = self.rl_model_predict(dataframe)
        dataframe['buy'] = (action == 1).astype('int')

        logger.info(f'{metadata["pair"]} - buy signal populated!')
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        logger.info(f'Populating sell signal for {metadata["pair"]}')
        action = self.rl_model_predict(dataframe)
        dataframe['sell'] = (action == 2).astype('int')
        logger.info(f'{metadata["pair"]} - sell signal populated!')
        return dataframe

    def rl_model_predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(np.zeros((len(dataframe), 1)))

        indicators = (
            dataframe[dataframe.columns[~dataframe.columns.isin(COLUMNS_FILTER)]]
            .fillna(0)
            .to_numpy()
        )
        indicators2 = dataframe[dataframe.columns[~dataframe.columns.isin(COLUMNS_FILTER)]].fillna(
            0
        )








        t1 = time.time()
        for window in range(self.window_size, len(dataframe)):
            start = window - self.window_size
            end = window
            observation = indicators[start:end]
            res, _ = predict.predict(observation, deterministic=True)
            output.loc[end] = res
        logger.info(f'Elapsed time default: {time.time() - t1}')

        return output

    def predict(self, x: pd.Series):
        logger.info(f'Predicting action for {x}, type: {type(x)}')
        return predict.predict(x, deterministic=True)


def normalize(data, min_value, max_value):
    """
    Given a list of numbers, normalize each number by subtracting the minimum and dividing by the
    maximum

    :param data: The data to be normalized
    :param min_value: the minimum value in the data
    :param max_value: The maximum value in the column
    :return: The normalized data
    """
    return (data - min_value) / (max_value - min_value)
