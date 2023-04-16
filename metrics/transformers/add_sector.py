if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import os
import pandas as pd

def get_sector() -> pd.DataFrame:
    """
    This function reads a CSV file containing stock tickers and corresponding sectors, and returns a DataFrame
    with ticker symbols and their corresponding sectors.

    :return: pd.DataFrame: DataFrame with ticker symbols and corresponding sectors
    """
    file_path = 'tickers.csv'
    absolute_path = os.path.abspath(file_path)
    df_sector = pd.read_csv(absolute_path)
    df_sector = df_sector[['Symbol','Sector']]
    df_sector = df_sector.rename(columns={'Symbol': 'ticker', 'Sector': 'sector'})
    return df_sector

@transformer
def transform(df) -> pd.DataFrame:
    """
      Transforms the input data frame by adding a 'sector' column based on ticker symbols.

      This function maps ticker symbols in the input data frame to their corresponding sector values
      using a reference data frame obtained from the 'get_sector()' function. The resulting 'sector' values
      are added as a new column to the input data frame. If a ticker symbol does not have a corresponding
      sector value in the reference data frame, it is filled with 'others'.

      Args:
          df (pd.DataFrame): The input data frame from upstream blocks

      Returns:
          pd.DataFrame: Transformed data frame with added 'sector' column
    """

    df_sector = get_sector()

    df['sector'] = df['ticker'].map(df_sector.set_index('ticker')['sector']).fillna("others")

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
