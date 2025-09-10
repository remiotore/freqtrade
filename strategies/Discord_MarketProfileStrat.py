# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from typing import Optional
from market_profile import MarketProfile

class MarketProfileStrat(IStrategy):
    minimal_roi = {
        "0":  1
    }
    stoploss = -0.01

    plot_config = {
        'main_plot': {
            'profile_poc': {'color': 'blue'},
            'profile_low': {'color': 'blue'},
            'profile_high': {'color': 'blue'},
            'profile_val': {'color': 'blue'},
            'profile_vah': {'color': 'blue'},
            'profile_bt': {'color': 'blue'},
        },
        'subplots': {
        }
    }
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # mmp
        dataframe = mp(dataframe=dataframe, mode="vol", tick_size=0.001)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


# https://github.com/bfolkens/py-market-profile/blob/master/src/market_profile/__init__.py
# from market_profile import MarketProfile
# mode: "vol" or "tpo"
#     "vol" - groupby volume
#     "tpo" - groupby close price
# tick_size: 0.001
def mp(dataframe: DataFrame, mode: str = "vol", tick_size: float = 0.001):
    # Create a copy of the dataframe and rename close to Close
    # This is needed for the MarketProfile indicator
    df = dataframe.copy()
    df.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)

    # Iterate in dataframe day by day
    for i in range(1, len(dataframe)):
        # Get the day
        day = dataframe.loc[i, "date"].date()
        # Get the day before
        day_before = dataframe.loc[i-1, "date"].date()

        # If the day is different from the day before
        if day == day_before:
            # Get index of first row of the day
            first_row = dataframe[dataframe["date"].dt.date == day].index.min()
            # Get the MarketProfile of the day
            mp = MarketProfile(df[first_row:i], mode=mode, tick_size=tick_size)
            mp_slice = mp[0:i-first_row]

            # Get all the values
            #ib_low, ib_high = mp_slice.initial_balance()
            #or_low, or_high = mp_slice.open_range()
            profile_low, profile_high = mp_slice.profile_range
            val, vah = mp_slice.value_area

            #dataframe.loc[i, "or_low"] = or_low
            #dataframe.loc[i, "or_high"] = or_high
            #dataframe.loc[i, "ib_low"] = ib_low
            #dataframe.loc[i, "ib_high"] = ib_high
            dataframe.loc[i, "profile_poc"] = mp_slice.poc_price
            dataframe.loc[i, "profile_low"] = profile_low
            dataframe.loc[i, "profile_high"] = profile_high
            dataframe.loc[i, "profile_val"] = val
            dataframe.loc[i, "profile_vah"] = vah
            dataframe.loc[i, "profile_bt"] = mp_slice.balanced_target
            #dataframe.loc[i, "lvn"] = mp_slice.low_value_nodes
            #dataframe.loc[i, "hvn"] = mp_slice.high_value_nodes

    return dataframe
