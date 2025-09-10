
import logging
import time
from datetime import timedelta
from functools import partial

import numpy as np  # noqa
import pandas as pd  # noqa
from freqtrade.strategy.interface import IStrategy
from keras import Sequential
from keras.models import load_model
from pandas import DataFrame

from time_series_to_gaf.cnn_model import create_cnn
from time_series_to_gaf.constants import REPO
from time_series_to_gaf.preprocess import quick_gaf, tensor_transform

logger = logging.getLogger(__name__)

COLUMNS_FILTER = [
    'date',
    'open',
    'close',
    'high',
    'low',
    'buy',
    'sell',
    'volume',
    'buy_tag',
    'exit_tag',
]

from scipy import stats

stats.zscore = partial(stats.zscore, nan_policy='omit')


class my_gym_cnn(IStrategy):









    minimal_roi = {"0": 100}

    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.017
    trailing_only_offset_is_reached = True

    timeframe = '1h'

    use_sell_signal = True

    process_only_new_candles = True

    startup_candle_count: int = 504

    models: dict[str, Sequential] = {}
    window_size = 504

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        model_map = {
            'BTC/USDT': ('20220403125044', 'RandomUniform.h5'),
            'ETH/USDT': ('20220403113648', 'none.h5'),
            'LTC/USDT': ('20220403120134', 'LecunUniform.h5'),
        }
        for pair, model_info in model_map.items():
            try:
                self.models[pair] = self.load_model(pair, *model_info)
            except Exception as e:
                raise RuntimeError(f'Could not load model: {e}') from e
            else:
                logger.info(f'Loaded model: {model_info}')



    def load_model(self, pair: str, time: str, model_name: str) -> Sequential:

        model_to_load = REPO / pair.replace('/', '_') / time / 'models' / model_name

        return load_model(model_to_load)

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


























        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        assert self.models.get(metadata['pair']) is not None, 'Model is not loaded.'
        logger.info(f'Populating buy signal for {metadata["pair"]}')
        action = self.rl_model_predict(dataframe, metadata['pair'])
        dataframe['buy'] = (action[0] > 0.50).astype('int')
        dataframe['sell'] = (action[1] > 0.50).astype('int')

        print(dataframe['buy'].value_counts(), 'buy signals')
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



        print(dataframe['sell'].value_counts(), 'sell signals')
        logger.info(f'{metadata["pair"]} - sell signal populated!')
        return dataframe

    def preprocess(self, indicators: pd.DataFrame):
        t1 = time.perf_counter()
        images = quick_gaf(indicators)[0]
        images = tensor_transform(images[-1])
        print('preprocess() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
        return images

    def rl_model_predict(self, dataframe: DataFrame, pair: str):
        action_output = pd.DataFrame(np.zeros((len(dataframe), 2)))





        indicators = dataframe.copy()[['date', 'open', 'close']]



        for window in range(self.window_size, len(dataframe), 24):
            start = window - self.window_size
            end = window
            observation = self.preprocess(indicators[start:end])
            t1 = time.perf_counter()
            res = self.models[pair].predict(observation)[0]
            print('model.predict() -> Elapsed time:', timedelta(seconds=time.perf_counter() - t1))
            action_output.loc[end] = res

        return action_output
















