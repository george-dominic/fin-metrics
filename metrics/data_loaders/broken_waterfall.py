from openbb_terminal.sdk import openbb
import pandas as pd
import os

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

# def get_ticker_list():

#     # read the wikipedia page that lists the current S&P 500 constituents
#     url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#     table = pd.read_html(url, header=0)[0]

#     # extract the ticker symbol column
#     tickers = table['Symbol'].tolist()

#     return tickers

@data_loader
def get_ratio_growth_stocks():
    count =0
    lst_stock_df = []
    for tick in get_ticker_list():
        df_growth_ratio = openbb.stocks.fa.growth(tick, limit = 20)
        df_growth_ratio = df_growth_ratio.T.drop('Period',axis=1)
        df_growth_ratio['Year'] = df_growth_ratio.index
        df_growth_ratio = df_growth_ratio.reset_index(drop=True)
        df_growth_ratio['Ticker'] = [tick for i in range(0,len(df_growth_ratio.index))]
        lst_stock_df.append(df_growth_ratio)
        count+=1
        if count == 3:
            break
    return pd.concat(lst_stock_df)

# print(get_ratio_growth_stocks())

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
