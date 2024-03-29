if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def target_encoding(df: pd.DataFrame, column, target) -> pd.DataFrame:
    """
    Target encode a categorical column in a DataFrame using the mean target value of each category.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the categorical column to encode.
        target (str): The name of the target column to use for encoding.

    Returns:
        pandas.DataFrame: The DataFrame with the encoded column.
    """
    # Encode the categorical column using label encoding
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])

    # Compute the mean target value for each category and replace the encoded values with the mean values
    mean_encode = df.groupby(column)[target].mean()
    df[column] = df[column].map(mean_encode)

    return df


def ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from the input DataFrame.

    Args:
    df (pandas.DataFrame): The input DataFrame containing the features.

    Returns:
    pandas.DataFrame: A new DataFrame with the additional created features.
    """
    # Copy the input DataFrame and remove the target column
    df_feature_creation = df.iloc[:,:-1]

    # Create new features using arithmetic combinations of existing features
    df_feature_creation["Revenue_growth/Ebitgrowth"] = df_feature_creation["Revenue_growth"]/df_feature_creation["Ebitgrowth"]
    df_feature_creation["Net_income_growth/Operating_cash_flow_growth"] = df_feature_creation["Net_income_growth"]/df_feature_creation["Operating_cash_flow_growth"]
    df_feature_creation["Gross_profit_growth/Revenue_growth"] = df_feature_creation["Gross_profit_growth"]/df_feature_creation["Revenue_growth"]
    df_feature_creation["Free_cash_flow_growth/Operating_cash_flow_growth"] = df_feature_creation["Free_cash_flow_growth"]/df_feature_creation["Operating_cash_flow_growth"]

    return df_feature_creation

def calc_sector_stats(df: pd.DataFrame, metric_col, sector_col) -> pd.DataFrame:
    """
    Calculates the mean and standard deviation of a financial metric
    based on the sector of each stock in a DataFrame.

    Parameters:
    df (Pandas DataFrame): the input DataFrame containing financial metrics data
    metric_col (str): the name of the column containing the financial metric of interest
    sector_col (str): the name of the column containing the sector of each stock

    Returns:
    df. Modifies the input DataFrame 'df' by adding new columns for the
    mean and standard deviation of the metric for each sector.
    """
    # Calculate mean and standard deviation for each sector
    sector_stats = df.groupby(sector_col)[metric_col].agg(['mean', 'std'])

    # Rename columns for clarity
    sector_stats.columns = ['Mean ' + metric_col, 'Std Dev ' + metric_col]

    # Add new columns to input DataFrame based on sector
    df = pd.merge(df, sector_stats, left_on=sector_col, right_index=True, how='left')

    return df

def sector_transformation(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Performs sector statistics calculation for certain features in the dataframe

    Parameters:
    df (Pandas DataFrame)

    Returns:
    df (Pandas DataFrame)
    '''
    df = calc_sector_stats(df, "Revenue_growth", "Sector")
    df = calc_sector_stats(df, "Ebitgrowth", "Sector")
    df = calc_sector_stats(df, "Operating_cash_flow_growth", "Sector")
    df = calc_sector_stats(df, "Net_income_growth", "Sector")
    return df

def generate_poly_features(df: pd.DataFrame, columns) -> pd.DataFrame:
    """
    Generates polynomial features of degree 2 for the specified columns in the input DataFrame.

    Args:
    df (pandas.DataFrame): The input DataFrame containing the original features.
    columns (list of str): The list of column names to generate polynomial features for.

    Returns:
    pandas.DataFrame: A new DataFrame containing the original features and the new polynomial features.
    """

    # Create a copy of the input DataFrame with only the specified columns
    df_subset = df[columns].copy()

    # Create a PolynomialFeatures object with degree=2
    poly = PolynomialFeatures(degree=2)

    # Transform the input feature matrix into a new polynomial feature matrix
    poly_features = poly.fit_transform(df_subset)

    # Create a list of column names for the new polynomial features
    poly_feature_names = poly.get_feature_names_out(df_subset.columns)

    # Create a new DataFrame with the original features and the new polynomial features
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

    # Concatenate the original DataFrame with the new polynomial features DataFrame
    df_concat = pd.concat([df, df_poly], axis=1)

    return df_concat

def call_poly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates polynomial features based on a list of columns in the input DataFrame.

    Args:
    df (pandas.DataFrame): The input DataFrame containing the columns to create polynomial features from.

    Returns:
    pandas.DataFrame: A new DataFrame containing the original columns and polynomial features.
    """

    # Specify the list of columns to create polynomial features from
    poly_features_lst = ['Revenue_growth','Ebitgrowth','Net_income_growth','Operating_cash_flow_growth','Gross_profit_growth','Free_cash_flow_growth']

    # Generate the polynomial features
    poly_df = generate_poly_features(df, poly_features_lst)

    # Remove duplicated columns
    poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()]

    return poly_df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the columns of a pandas DataFrame according to a given dictionary of column names.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame whose columns will be renamed.
    column_map : dict
        A dictionary of old column names mapped to new column names.

    Returns:
    --------
    pandas DataFrame
        The DataFrame with renamed columns.
    """

    column_map = {
    'Free_cash_flow_growth^2': 'Free_cf_growth_squared',
    }
    # Rename the columns using the dictionary
    df = df.rename(columns=column_map)

    # Return the updated DataFrame
    return df

def retrieve_columns():
    # open the text file containing column names
    with open('columns.txt', 'r') as f:
    # read the column names from the file and create a list
        columns = f.read().splitlines()
        return columns
    
@transformer
def transform(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Transforms the input DataFrame by encoding categorical features, creating new features,
    generating polynomial features, selecting the top features, and returning the resulting DataFrame.

    Args:
    df (pandas.DataFrame): The input DataFrame to transform.

    Returns:
    pandas.DataFrame: The transformed DataFrame.
    """
    # Encode categorical feature 'Sector' using target encoding
    data_encoded = target_encoding(df, 'Sector', 'Stock_Percent_Change')

    # Create new features using the encoded DataFrame
    new_ratio_features = ratio_features(data_encoded)

    #Adding new sector columns based on their mean and stdev
    sector_transform = sector_transformation(new_ratio_features)

    # Add the original 'Stock_Percent_Change' feature back to the DataFrame
    new_poly_features = call_poly(sector_transform)

    #stock percent added back
    df_fin_metric = pd.concat([new_poly_features, data_encoded['Stock_Percent_Change']], axis=1)

    # Select top features using different methods and store them in separate lists
    # rf_features = select_features_rf(df_fin_metric, 'Stock_Percent_Change', 40)
    # sklearn_feature = select_k_best_features(df_fin_metric, 'Stock_Percent_Change', k=40)
    # xgboost_feature = xgboost_feature_selection(df_fin_metric, 'Stock_Percent_Change', 40)

    # # Get the common features selected by all three methods and append 'Stock_Percent_Change'
    # top_features = list(set(rf_features) & set(sklearn_feature) & set(xgboost_feature))
    # top_features.append('Stock_Percent_Change')

    # Create the final DataFrame using only the top selected features
    top_features = retrieve_columns()
    df_fin_metric_final = df_fin_metric[top_features]

    #renaming columns to comply with big query column name conventions
    df_fin_metric_final = rename_columns(df_fin_metric_final)

    return df_fin_metric_final


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
