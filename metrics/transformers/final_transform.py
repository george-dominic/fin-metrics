if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pandas as pd

@transformer
def transform(df : pd.DataFrame, df2 : pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Concatenate two dataframes along their columns,
    and return the concatenated dataframe.

    Args:
        df: The first input dataframe.
        df2: The second input dataframe.

    Returns:
        The concatenated dataframe.
    """
    return pd.concat([df,df2], axis = 1)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
