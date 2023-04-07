if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pandas as pd
import os


# Create a dataframe with ticker names
def read_ticker_industry() -> pd.DataFrame:
    """
    Reads a CSV file containing ticker symbols and industry names,
    and returns a dataframe containing the tickers and their industries.
    """
    # specify the relative or absolute path of the file
    file_path = 'tickers.csv'
    # get the absolute path of the file
    absolute_path = os.path.abspath(file_path)
    df = pd.read_csv(absolute_path)
    df = df.rename(columns={'Symbol': 'Ticker'})
    df = df[['Ticker', 'Sector']]

    return df
    

@transformer
def transform(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Merge the input dataframe with the ticker-industry mapping,
    and return a dataframe with only the 'Ticker' and 'Industry' columns.
    """
    # Specify your transformation logic here
    merged_df = pd.merge(df, read_ticker_industry(), on='Ticker', how='left')
    merged_df = merged_df[['Ticker','Sector']]

    merged_df['Sector'] = merged_df['Sector'].fillna(value='Miscellaneous')

    return merged_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
