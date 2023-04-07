if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

def handle_k_m(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame as input and processes any string values in the DataFrame that end with "K" or "M" to 
    convert them to floats. It then returns a new DataFrame with the processed values, where each column of the new 
    DataFrame corresponds to a column of the original DataFrame and has had its string values converted to floats where 
    appropriate.
    
    Args:
        df: A pandas DataFrame containing columns with string values that may need to be converted to floats.
    
    Returns:
        A new pandas DataFrame with the same columns as the input DataFrame, but with any string values that end with "K" 
        or "M" converted to floats where appropriate.
    """
    ticker_drop = df.drop("Ticker",axis = 1)
    temp = {}
    for key, value in ticker_drop.items():
        lst = []
        for val in value.to_list():
            try:
                val_float = float(val)
            except ValueError:
                if val[-1].upper() == "K":
                    val_float = float(val[:-1]) * 1000
                elif val[-1].upper() == "M":
                    val_float = float(val[:-1]) * 1000000
            lst.append(val_float)
        temp[key] = lst
    return pd.DataFrame(temp)


def remove_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame as input and drops columns due to huge percentage of zeros

    Args:
        df: A pandas DataFrame containing data to be transformed.
    
    Returns:
        A new pandas DataFrame containing the transformed data.

    """
    cols = ["Rdexpense_growth","Operating_income_growth","Epsdiluted_growth",
        "Epsgrowth","Weighted_average_shares_diluted_growth","Book_valueper_share_growth","Year"]
    return df.drop(cols, axis = 1)


def handle_skew(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame as input and applies a Yeo-Johnson power transform to the data to reduce skewness. 
    It then returns a new DataFrame containing the transformed data.
    
    Args:
        df: A pandas DataFrame containing data to be transformed.
    
    Returns:
        A new pandas DataFrame containing the transformed data.
    """
    
    # Define the PowerTransformer with the Yeo-Johnson method
    pt = PowerTransformer(method='yeo-johnson')

    # Fit and transform the data using the PowerTransformer
    data_transformed = pt.fit_transform(df)

    # Convert the transformed data to a new DataFrame
    return pd.DataFrame(data_transformed, columns=df.columns)


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and standardizes all features except the last column using the 
    StandardScaler from scikit-learn. It then returns a new DataFrame containing the standardized data with the last 
    column included.
    
    Args:
        df: A pandas DataFrame containing the data to be standardized.
    
    Returns:
        A new pandas DataFrame containing the standardized data with the last column included.
    """
    # Define the StandardScaler
    scaler = StandardScaler()

    # Scale the data excluding the last column
    scaled_data = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1], index=df.index)

    # Add the last column to the scaled data
    scaled_data[df.columns[-1]] = df.iloc[:, -1]

    return scaled_data


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and applies Winsorization to the data to handle outliers.
    Specifically, it replaces all data points below the 5th percentile with the value at the 5th percentile and 
    all data points above the 95th percentile with the value at the 95th percentile. The function then returns a 
    new DataFrame containing the Winsorized data with the last column included.
    
    Args:
        df: A pandas DataFrame containing the data to be Winsorized.
    
    Returns:
        A new pandas DataFrame containing the Winsorized data with the last column included.
    """
    # Define the lower and upper percentiles
    lower_percentile = 5
    upper_percentile = 95

    # Choose all columns except the last one
    cols = df.columns[:-1]

    # For each column, find the values at the chosen lower and upper percentiles
    lower_limits = df[cols].apply(lambda x: np.percentile(x, lower_percentile))
    upper_limits = df[cols].apply(lambda x: np.percentile(x, upper_percentile))

    # Replace all data points below the lower percentile with the value at the lower percentile
    # Replace all data points above the upper percentile with the value at the upper percentile
    data_winsorized = df[cols].apply(lambda x: np.clip(x, lower_limits[x.name], upper_limits[x.name]))

    # Concatenate the Winsorized columns with the last column
    return pd.concat([data_winsorized, df.iloc[:, -1]], axis=1)

def add_sectors(df:pd.DataFrame, df_raw:pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Sector' column to the given DataFrame by mapping Ticker symbols to their respective sectors from a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to which the 'Sector' column needs to be added.

    Returns:
        pandas.DataFrame: A new DataFrame with the 'Sector' column added.
    """
    file_path = 'tickers.csv'
    absolute_path = os.path.abspath(file_path)
    df_sector = pd.read_csv(absolute_path)
    sector_dic = df_sector.set_index('Symbol')['Sector']
    df.insert(len(df.columns)-1, 'Sector', df_raw['Ticker'].map(sector_dic).fillna('others'))
    return df

@transformer
def transform(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    This function takes a pandas DataFrame as input and applies a series of preprocessing steps to the data.
    
    Args:
        df: A pandas DataFrame containing the data to be preprocessed.
    
    Returns:
        A new pandas DataFrame containing the preprocessed data.
    """
    # df_raw = df
    # Step 1: Handle values with "K" and "M" characters
    df1 = handle_k_m(df)

    # Step 2: Remove columns with high percentage of zeros
    df2 = remove_col(df1)

    # Step 3: Handle skewness with Yeo-Johnson power transform
    df3 = handle_skew(df2)

    # Step 4: Standardize the data
    df4 = standardize_data(df3)

    # Step 5: Handle outliers with Winsorization
    df5 = handle_outliers(df4)

    # Step 6: Add Sectors against the tickers
    df6 = add_sectors(df5,df)

    # Return the transformed DataFrame
    return df6



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
