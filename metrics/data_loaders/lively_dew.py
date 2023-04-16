if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import io
import pandas as pd
from openbb_terminal.sdk import openbb
import os
from typing import List
import random
import time

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

def get_news(ticker: str) -> pd.DataFrame:
    news = openbb.stocks.ba.cnews(ticker,"2022-01-01","2022-12-31")
    # Extract headlines and summaries from news articles
    timestamp = [article['datetime'] for article in news]
    headlines = [article['headline'] for article in news]
    summaries = [article['summary'] for article in news]
    # Create a Pandas DataFrame
    news_df = pd.DataFrame({'timestamp' : timestamp, 'headline': headlines, 'summary': summaries})
    news_df['ticker'] = [ticker for i in range(0, news_df.shape[0])]
    return news_df

@data_loader
def load_news(*args, **kwargs) -> pd.DataFrame:
    """
    Template for loading data from API
    """
    api_limit = 10
    counter = 0
    run_count = 0

    #checking if last processed counter exists, if yes it picks it up
    if os.path.exists('news_counter.txt'):
        with open('news_counter.txt', 'r') as f:
            counter = int(f.read().strip())

    lst_news_df = []

    for ticker in read_ticker_list()[counter:]:
        print("Currently Processing :",counter)
        time.sleep(2 + 0.5 * random.random())
        df_news = get_news(ticker)
        if df_news.empty:
            continue
        lst_news_df.append(df_news)

        counter += 1
        run_count += 1

        if run_count == api_limit:
            #writing counter with the last run to resume run from there
            with open('news_counter.txt', 'w') as f:
                f.write(str(counter))
            print("Closing API limit, Successful Run")
            break

    return pd.concat(lst_news_df)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
