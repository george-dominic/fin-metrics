from openbb_terminal.sdk import openbb
import os
import pandas as pd
import yfinance as yf

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def get_ticker_list():

    # specify the relative or absolute path of the file
    file_path = 'tickers.csv'
    # get the absolute path of the file
    absolute_path = os.path.abspath(file_path)
    df = pd.read_csv(absolute_path)
    return df['Symbol'].to_list()

def get_pct_change_ticker(ticker):

    start_date = "2002-01-01"
    end_date = "2022-12-31"

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    yearly_prices = stock_data["Adj Close"].resample("Y").last()
    year_range = pd.date_range(start=start_date, end=end_date, freq='Y')
    yearly_prices = yearly_prices.reindex(year_range)
    yearly_prices = yearly_prices.ffill()
    yearly_returns = yearly_prices.pct_change().fillna(0)
    yearly_returns.drop(yearly_returns.index[0], inplace=True)

    return yearly_returns.to_list()

@data_loader
def get_ratio_growth_stocks():

    count = 0
    lst_stock_df = []
    for tick in get_ticker_list():
        df_growth_ratio = openbb.stocks.fa.growth(tick, limit=20)
        df_growth_ratio = df_growth_ratio.T.drop('Period', axis=1)
        df_growth_ratio['Year'] = df_growth_ratio.index
        df_growth_ratio = df_growth_ratio.reset_index(drop=True)
        df_growth_ratio['Stock Percent Change'] = get_pct_change_ticker(tick)
        df_growth_ratio['Ticker'] = [
            tick for i in range(0, len(df_growth_ratio.index))]
        lst_stock_df.append(df_growth_ratio)
        count += 1
        if count == 3:
            break
    return pd.concat(lst_stock_df)

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'