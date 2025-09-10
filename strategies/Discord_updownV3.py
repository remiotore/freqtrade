# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from indicators import *
from pandas import DataFrame

# ================== SUMMARY METRICS ==================
# | Metric                | Value                     |
# |-----------------------+---------------------------|
# | Backtesting from      | 2021-01-11 00:00:00       |
# | Backtesting to        | 2021-06-22 06:30:00       |
# | Max open trades       | 2                         |
# |                       |                           |
# | Total trades          | 423                       |
# | Starting balance      | 10000.000 USDT            |
# | Final balance         | 10210686.336 USDT         |
# | Absolute profit       | 10200686.336 USDT         |
# | Total profit %        | 102006.86%                |
# | Trades per day        | 2.61                      |
# | Avg. stake amount     | 100.000 USDT              |
# | Total trade volume    | 42300.000 USDT            |
# |                       |                           |
# | Best Pair             | BNBDOWN/USDT 10190633.01% |
# | Worst Pair            | BNBUP/USDT -137.17%       |
# | Best trade            | BNBDOWN/USDT 10190651.61% |
# | Worst trade           | BNBUP/USDT -10.18%        |
# | Best day              | 10200846.683 USDT         |
# | Worst day             | -26.805 USDT              |
# | Days win/draw/lose    | 65 / 4 / 94               |
# | Avg. Duration Winners | 5:14:00                   |
# | Avg. Duration Loser   | 6:43:00                   |
# | Zero Duration Trades  | 0.00% (0)                 |
# | Rejected Buy signals  | 0                         |
# |                       |                           |
# | Min balance           | 9915.997 USDT             |
# | Max balance           | 10210825.191 USDT         |
# | Drawdown              | 156.61%                   |
# | Drawdown              | 156.763 USDT              |
# | Drawdown high         | 10200825.191 USDT         |
# | Drawdown low          | 10200668.428 USDT         |
# | Drawdown Start        | 2021-04-17 02:15:00       |
# | Drawdown End          | 2021-05-31 08:15:00       |
# | Market change         | 526.65%                   |
# =====================================================

def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

class updownV3(IStrategy):
    startup_candle_count: int = 400
    timeframe = '15m'

    minimal_roi = {
        "0": 0.10
    }

    stoploss = -0.10
    trailing_stop = False
    process_only_new_candles = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for c in ['buy', 'sell']:
            if not c in dataframe.columns:
                dataframe.loc[:, c] = 0

        coin = metadata['pair'].split('/')[0]
        stake = metadata['pair'].split('/')[1]
        base =  coin.replace("UP","")
        base =  base.replace("DOWN","")

        informative_base = self.dp.get_pair_dataframe(pair=base+"/"+stake,
                                                  timeframe=self.timeframe)

        informative_base['typical'] = qtpylib.typical_price(informative_base)
        informative_base['sslDown'], informative_base['sslUp'] = SSLChannels(informative_base,14)
        dataframe = pd.merge(dataframe, informative_base, left_on='date', right_on='date', how='left',suffixes=(None,"_base"))
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        coin = metadata['pair'].split('/')[0]
        conditions = []

        if "UP" in coin:
            conditions = [
                (qtpylib.crossed_below(dataframe['sslUp'], dataframe['sslDown'])),
            ]
        if "DOWN" in coin:
            conditions = [
               (qtpylib.crossed_above(dataframe['sslUp'], dataframe['sslDown'])),
            ]
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        coin = metadata['pair'].split('/')[0]
        conditions = []

        if "DOWN" in coin:
            conditions = [
               (qtpylib.crossed_below(dataframe['sslUp'], dataframe['sslDown'])),
            ]
        if "UP" in coin:
            conditions = [
               (qtpylib.crossed_above(dataframe['sslUp'], dataframe['sslDown'])),
            ]

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1
        return dataframe
