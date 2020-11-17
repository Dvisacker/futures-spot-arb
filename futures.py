if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "scripts"

import ccxt
import time
import logging
import pandas as pd
import numpy as np
import dateutil
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.formula.api as sm

from datetime import timedelta, datetime

print('CCXT version:', ccxt.__version__)  # requires CCXT version > 1.20.31

ftx = ccxt.ftx()

ftx.load_markets()

def plot_residuals(df):
    # months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df.residual, label="Residuals")
    # ax.xaxis.set_major_locator(months)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Date')
    plt.ylabel('')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df.res)
    plt.show()

def get_binance_funding_rate(symbol='BTCUSDT', start='01/09/2020', end='14/09/2020'):
    start_datetime = datetime.strptime(start, '%d/%m/%Y')
    end_datetime = datetime.strptime(end, '%d/%m/%Y')
    start_timestamp = time.mktime(start_datetime.timetuple()) * 1000
    end_timestamp = time.mktime(end_datetime.timetuple()) * 1000
    data = ftx.fapiPublic_get_fundingrate(params={'symbol': 'BTCUSDT', 'startTime': int(start_timestamp), 'endTime': int(end_timestamp) })
    parser = lambda x : { 'timestamp': x['fundingTime'] / 1000, 'funding': float(x['fundingRate']), 'date': datetime.fromtimestamp(x['fundingTime'] / 1000) }
    parsed_data = list(map(parser, data))
    df = pd.DataFrame(parsed_data, columns=['timestamp', 'funding', 'date'])
    df.set_index('date', inplace=True)
    return df


def get_ftx_funding_rate(symbol='ETH-PERP', start='01/09/2020', end='14/09/2020'):
    start_datetime = datetime.strptime(start, '%d/%m/%Y')
    end_datetime = datetime.strptime(end, '%d/%m/%Y')
    start_timestamp = time.mktime(start_datetime.timetuple()) * 1000
    end_timestamp = time.mktime(end_datetime.timetuple()) * 1000
    data = ftx.public_get_funding_rates(params={'future': 'ETH-PERP', 'startTime': start_timestamp, 'endTime': end_timestamp })
    parser = lambda x : { 'amount': float(x['rate']), 'date': dateutil.parser.parse(x['time']) }
    parsed_data = list(map(parser, data['result']))
    df = pd.DataFrame(parsed_data, columns=['amount', 'date'])
    df.set_index('date', inplace=True)
    return df

def get_ftx_ohlcv(symbol='ETH-PERP', timeframe='1m', start='01/09/2020', end='14/09/2020'):
    delta = {
      '1m': timedelta(hours=12),
      '5m': timedelta(days=2, hours=12),
      '1h': timedelta(days=30),
      '1d': timedelta(days=365)
    }[timeframe]

    limit = {
      '1m': 720,
      '5m': 720,
      '15m': 720,
      '1h': 720,
      '1d': 365
    }[timeframe]

    start_datetime = datetime.strptime(start, '%d/%m/%Y')
    end_datetime = datetime.strptime(end, '%d/%m/%Y')
    start_timestamp = time.mktime(start_datetime.timetuple()) * 1000
    end_timestamp = time.mktime(end_datetime.timetuple()) * 1000
    current_datetime = start_datetime

    ohlcv = []
    while current_datetime < end_datetime:
      while True:
        try:
          since = time.mktime(current_datetime.timetuple()) * 1000
          data = ftx.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
          current_datetime += delta
          parser = lambda x : { 'open': x[1], 'high': x[2], 'low': x[3], 'close': x[4], 'date': datetime.fromtimestamp(x[0] / 1000) }
          parsed_ohlcv = list(map(parser, data))
          ohlcv += parsed_ohlcv
        except Exception as e:
          if e.__class__.__name__ == "DDoSProtection":
            logging.warning('Download is being rate-limited. Retrying in 2 seconds')
            time.sleep(2)
            continue
          else:
            logging.error(e)
            break

        break

    df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated()]
    return df


def compute_twap(df):
    df['weight'] = np.arange(start=0, stop=len(df))
    df['vwap'] = (df.weight * (df.open + df.high + df.low + df.close) / 4).cumsum() / df.weight.cumsum()
    return df
  
def get_crossunders(x, val):
  # col is price column
  crit1 = x.shift(1) > val
  crit2 = x < val
  return (crit1) & (crit2)


def get_crossovers(x, val):
  # col is price column
  crit1 = x.shift(1) < val
  crit2 = x > val
  return (crit1) & (crit2)

def get_result_df(ethusd, ethusd_futures, funding):
  ethusd_with_vwap = compute_twap(ethusd)

  df = pd.DataFrame()
  # Here we work with candles and close prices. A beter way to do this would be for example
  # 1) use bid and asks to compute the spread
  # 2) use L3 orderbook to take into account slippage, imperfect execution
  df['spot'] = ethusd.close
  df['future'] = ethusd_futures.close
  df['spread'] = ethusd_futures.close - ethusd.close
  df['spread_lagged_1'] = df.spread.shift(1)
  
  
  # We verify that the spread is stationary. Is pvalue is small (<0.05 for example), the series is stationary/mean-reverting.
  ad_result = ts.adfuller(df.spread, 1)
  print('ADF Statistic: %f' % ad_result[0])
  print('p-value: %f' % ad_result[1])
  print('Critical Values:')
  
  # We fit the historical to an OU process.
  # TODO train the model on previous historical data.  And verifies that it fits with current data
  # S_t - S_(t-1) = a + b * S_(t-1)
  # S_t = a + (1 + b) * S_(t-1)
  ols_result = sm.ols(formula='spread ~ spread_lagged_1', data=df).fit()
  a = ols_result.params[0]
  b = ols_result.params[1] - 1
  spread_mean = - a / b
  print(ols_result.summary())
  
  # We compute the spread crossunders to find the average time to mean reversion
  # This can be improved, for example finding average time to mean reversion from a certain spread value
  df['cross_under'] = get_crossunders(df.spread, spread_mean)
  df['cross_over'] = get_crossovers(df.spread, spread_mean)
  cross_df = pd.DataFrame(df[df.cross_under].cross_under | df[df.cross_over].cross_over)
  cross_df.reset_index(inplace=True)
  cross_timedeltas = cross_df.date.diff()
  mean_reversion_time_avg = cross_timedeltas.mean()
  mean_reversion_time_std = cross_timedeltas.std()
  
  
  # Simplification: https://help.ftx.com/hc/en-us/articles/360027946571-Funding
  # I'm unsure of the exact formula for ftx. The idea would be to use E(Funding) based on the
  # current value of the twap to decide or not whether it's profitable to enter a trade.
  # Alternatively we could use historical funding rates to decide whether to determine E(Funding)
  df['weight'] = np.arange(start=0, stop=len(df))
  df['funding'] = funding['amount']
  df['twap'] = (df.weight * (df.future - df.spot)).cumsum() / df.weight.cumsum()
  df.set_index(ethusd.index, inplace=True)

  return df

ethusd = get_ftx_ohlcv('ETH/USDT')
ethusd_futures = get_ftx_ohlcv('ETH-PERP')
funding = get_ftx_funding_rate('ETH-PERP')
df = get_result_df(ethusd, ethusd_futures, funding)

df.head()









# cadf = ts.adfuller(df['res'])
# get_ftx_funding_rate()
# df = get_ftx_ohlcv()
# import ipdb; ipdb.set_trace()
# get_ftx_ohlcv()