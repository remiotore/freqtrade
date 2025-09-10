class m(IStrategy):
    INTERFACE_VERSION: int = 3

    minimal_roi = {
        "0":  0.05
    }
    stoploss = -0.032
    timeframe = '5m'

    process_only_new_candles = False

    use_exit_signal = False
    exit_profit_only = True


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,  # type: ignore
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        stake = proposed_stake / 2
        return stake

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['ema20'], dataframe['ema50']) &
                (dataframe['ha_close'] > dataframe['ema20']) &
                (dataframe['ha_open'] < dataframe['ha_close'])  # green bar
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:







        return dataframe