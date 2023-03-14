from openbb_terminal.sdk import openbb
import os
from typing import List
import pandas as pd
import yfinance as yf

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def read_ticker_list() -> List[str]:
    """
    Reads a list of ticker symbols from a CSV file.

    Returns:
        A list of ticker symbols.
    """
    # specify the relative or absolute path of the file
    file_path = 'tickers.csv'
    # get the absolute path of the file
    absolute_path = os.path.abspath(file_path)
    df = pd.read_csv(absolute_path)
    return df['Symbol'].tolist()

def get_stock_returns(ticker: str, years: int) -> List[float]:
    """
    Calculates the percent change of a stock's price over the last `years` years.

    Args:
        ticker: The stock ticker symbol.
        years: The number of years to calculate the percent change over.

    Returns:
        A list of yearly stock returns as floats.
    """
    start_date = "2002-01-01"
    end_date = "2022-12-31"
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data is None:
        print(f"Error: Could not retrieve data for {ticker}.")
        return []
    yearly_prices = stock_data["Adj Close"].resample("Y").last().ffill()
    yearly_returns = yearly_prices.pct_change().fillna(0)[1:]
    return yearly_returns.tolist()[-years:]

def get_growth_ratios(ticker: str, years: int) -> pd.DataFrame:
    """
    Retrieves the financial growth ratios for a stock over the last `years` years.

    Args:
        ticker: The stock ticker symbol.
        years: The number of years to retrieve the growth ratios for.

    Returns:
        A DataFrame containing the growth ratios for the stock.
    """
    df = openbb.stocks.fa.growth(ticker, limit=years)
    if len(df) < 5:
        print(f"Skipping {ticker} due to insufficient data")
        return pd.DataFrame()
    df = df.T.drop('Period', axis=1)
    df['Year'] = df.index
    df = df.reset_index(drop=True)
    df['Ticker'] = [ticker for i in range(0, len(df))]
    return df

@data_loader
def get_ratio_growth_stocks() -> pd.DataFrame:
    """
    Retrieves the financial growth ratios and stock returns for a list of ticker symbols.

    Returns:
        A DataFrame containing the growth ratios and stock returns for the tickers.
    """
    api_limit = 10
    counter = 0
    run_count = 0

    if os.path.exists('last_processed_counter.txt'):
        with open('last_processed_counter.txt', 'r') as f:
            counter = int(f.read().strip())

    lst_stock_df = []

    for ticker in read_ticker_list()[counter:]:
        df_growth_ratio = get_growth_ratios(ticker, 20)
        if df_growth_ratio.empty:
            continue
        stock_returns = get_stock_returns(ticker, len(df_growth_ratio))
        if not stock_returns:
            print(f"Insufficient data for {ticker} to calculate percent change")
            continue
        df_growth_ratio['Stock Percent Change'] = stock_returns
        lst_stock_df.append(df_growth_ratio)

        counter += 1
        run_count += 1

        if run_count == api_limit-1:
            with open('last_processed_counter.txt', 'w') as f:
                f.write(str(counter))
            print("Closing API limit, Successful Run")
            break

    return pd.concat(lst_stock_df)

    
    
@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'