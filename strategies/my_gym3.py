
import logging
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import informative
from freqtrade.strategy.interface import IStrategy
from lazyft.strategy import load_strategy
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
    'buy',
    'sell',
    'volume',
    'buy_tag',
    'exit_tag',
]

from scipy import stats

stats.zscore = partial(stats.zscore, nan_policy='omit')


class my_gym3(IStrategy):









    minimal_roi = {"0": 100}

    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.017
    trailing_only_offset_is_reached = True

    timeframe = '15m'

    use_sell_signal = True

    process_only_new_candles = False

    startup_candle_count: int = 200

    model = None
    window_size = None

    timeperiods = [7, 14, 21]
    percent_of_balance_dict = {}

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.model = None
        try:





            model_file = Path(
                'models/best_model_SagesGym2_SagesFreqtradeEnv_A2C_20220321_132443.zip'
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

    @informative('4h', 'BTC/{stake}')
    @informative('2h', 'BTC/{stake}')
    @informative('1h', 'BTC/{stake}')
    def populate_indicators_btc_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe[f'rsi'] = stats.zscore(ta.RSI(dataframe['close'], timeperiod=30))
        dataframe['top'] = stats.zscore(dataframe['close'].rolling(window=30).max())
        return dataframe

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







        dataframe[f'rsi'] = stats.zscore(ta.RSI(dataframe['close']))

        dataframe['ao'] = stats.zscore(
            pta.ao(dataframe['high'], dataframe['low'], fast=12, slow=26)
        )

        macd, macdsignal, macdhist = ta.MACD(dataframe['close'])
        dataframe['macd'] = stats.zscore(macd)
        dataframe['macdsignal'] = stats.zscore(macdsignal)
        dataframe['macdhist'] = stats.zscore(macdhist)

        dataframe['aroonup'], dataframe['aroondown'] = stats.zscore(
            ta.AROON(dataframe['high'], dataframe['low'], timeperiod=25)
        )
        dataframe['current_price'] = stats.zscore(dataframe['close'])
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        assert self.model is not None, 'Model is not loaded.'
        logger.info(f'Populating buy signal for {metadata["pair"]}')
        action = self.rl_model_predict(dataframe, metadata['pair'])
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
        action = self.rl_model_predict(dataframe, metadata['pair'])
        dataframe['sell'] = (action == 2).astype('int')

        print(dataframe['sell'].value_counts(), 'sell signals')
        logger.info(f'{metadata["pair"]} - sell signal populated!')
        return dataframe

    def rl_model_predict(self, dataframe: DataFrame, pair: str):
        action_output = pd.DataFrame(np.zeros((len(dataframe), 1)))


        indicators = dataframe.copy()
        for c in COLUMNS_FILTER:

            indicators = indicators.drop(columns=[col for col in indicators.columns if c in col])
        indicators = indicators.fillna(0).to_numpy()



        for window in range(self.window_size, len(dataframe)):
            start = window - self.window_size
            end = window
            observation = indicators[start:end]
            res, _ = predict.predict(observation, deterministic=True)
            action, percent_of_balance = res
            if action == 1:
                self.percent_of_balance_dict[end] = max(percent_of_balance, 1)
            action_output.loc[end] = action

        return action_output

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        entry_tag: Optional[str],
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        pob = self.percent_of_balance_dict[last_candle.name.astype('int')]
        if pob > 0:
            return pob / 10 * self.wallets.get_available_stake_amount()


def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)


def rolling_z_score(data: pd.Series, window):

    r = data.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (data - m) / s
    return z
