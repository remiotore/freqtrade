

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO


class FreqGym_rscaler_2(IStrategy):






    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.017
    trailing_only_offset_is_reached = True

    ticker_interval = "5m"

    use_sell_signal = True

    process_only_new_candles = False

    startup_candle_count: int = 200

    model = None
    window_size = None

    try:
        model = PPO.load(
            "models/best_model_SimpleROIEnv_PPO_20211028_104631"
        )  # Note: Make sure you use the same policy as the one used to train
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

        dataframe["plus_dm"] = ta.PLUS_DM(dataframe)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)

        dataframe["minus_dm"] = ta.MINUS_DM(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)

        dataframe["ao"] = qtpylib.awesome_oscillator(dataframe)

        dataframe["uo"] = ta.ULTOSC(dataframe)

        dataframe["ewo"] = EWO(dataframe, 50, 200)











        for period in self.timeperiods:

            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)

            dataframe[f"wr_{period}"] = williams_r(dataframe, timeperiod=period)

            dataframe[f"cci_{period}"] = ta.CCI(dataframe, timeperiod=period)

            dataframe[f"rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)

            rsi = 0.1 * (dataframe[f"rsi_{period}"] - 50)
            dataframe[f"fisher_rsi_{period}"] = (np.exp(2 * rsi) - 1) / (
                np.exp(2 * rsi) + 1
            )

            dataframe[f"fisher_rsi_norma_{period}"] = 50 * (
                dataframe[f"fisher_rsi_{period}"] + 1
            )

            aroon = ta.AROON(dataframe, timeperiod=period)
            dataframe[f"aroonup_{period}"] = aroon["aroonup"]
            dataframe[f"aroondown_{period}"] = aroon["aroondown"]
            dataframe[f"aroonosc_{period}"] = ta.AROONOSC(dataframe, timeperiod=period)

            dataframe[f"cmo_{period}"] = ta.CMO(dataframe, timeperiod=period)

            dataframe[f"mfi_{period}"] = ta.MFI(dataframe, timeperiod=period)









        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        action = self.rl_model_predict(dataframe)
        dataframe["buy"] = (action == 1).astype("int")

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        action = self.rl_model_predict(dataframe)
        dataframe["sell"] = (action == 2).astype("int")

        return dataframe

    def rl_model_predict(self, dataframe):
        output = pd.DataFrame(np.zeros((len(dataframe), 1)))
        indicators = (
            dataframe[
                dataframe.columns[
                    ~dataframe.columns.isin(
                        [
                            "date",
                            "open",
                            "close",
                            "high",
                            "low",
                            "volume",
                            "buy",
                            "sell",
                            "buy_tag",
                        ]
                    )
                ]
            ]
            .fillna(0)
            .to_numpy()
        )

        for window in range(self.window_size, len(dataframe)):
            start = window - self.window_size
            end = window
            observation = indicators[start:end]
            res, _ = self.model.predict(observation, deterministic=True)
            output.loc[end] = res

        return output


def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df["close"] * 100
    return smadif


def williams_r(dataframe: DataFrame, timeperiod: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
    of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
    Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
    of its recent trading range.
    The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=timeperiod).max()
    lowest_low = dataframe["low"].rolling(center=False, window=timeperiod).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{timeperiod} Williams %R",
    )

    return WR * -100
