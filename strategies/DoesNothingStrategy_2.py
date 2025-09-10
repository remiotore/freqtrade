
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional
from functools import reduce
from pandas import DataFrame, Series


import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
import time

logger = logging.getLogger(__name__)


class DoesNothingStrategy_2(IStrategy):
    """

    author@: Gert Wohlgemuth

    just a skeleton

    """



    minimal_roi = {
        "0": 1000
    }

    stoploss = -1

    timeframe = '1d'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe
