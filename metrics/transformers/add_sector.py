if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import os
import pandas as pd

def get_sector():
    file_path = 'tickers.csv'
    absolute_path = os.path.abspath(file_path)
    df_sector = pd.read_csv(absolute_path)
    df_sector = df_sector[['Symbol','Sector']]
    df_sector = df_sector.rename(columns={'Symbol': 'ticker', 'Sector': 'sector'})
    return df_sector

@transformer
def transform(df):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        args: The input variables from upstream blocks

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    df_sector = get_sector()

    df = pd.merge(df, df_sector, on='ticker', how='left')

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
