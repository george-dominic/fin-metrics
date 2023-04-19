if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import h2o
from h2o.automl import H2OAutoML
h2o.init()

def ml_model(df):
    """
    This function trains an H2O AutoML model on the input pandas dataframe, and returns the best model.

    Args:
        df (pandas.DataFrame): The input dataframe with the features and target variable.

    Returns:
        An H2O model object that represents the best model trained by the H2O AutoML.

    """
    df_h2o = h2o.H2OFrame(df)
    aml = H2OAutoML(max_runtime_secs = 60, exclude_algos = ["StackedEnsemble", "DeepLearning"], verbosity="info", seed = 10, nfolds = 10, project_name = "fds")
    aml.train(y = 'Stock_Percent_Change', training_frame = df_h2o )
    print(aml.leaderboard.head())
    return aml.leader

def make_predictions(model, df):
    """
    This function uses the input H2O model to make predictions on the input pandas dataframe, and returns a pandas dataframe with the original data and the predictions.

    Args:
        model (H2O model object): The trained H2O model to use for making predictions.
        df (pandas.DataFrame): The input dataframe with the features.

    Returns:
        A pandas dataframe with the original data and the predictions appended as a new column.

    """
    test_data_h2o = h2o.H2OFrame(df)
    preds = model.predict(test_data_h2o)
    test_data_h2o = test_data_h2o.cbind(preds)
    test_data_h2o = test_data_h2o.as_data_frame()
    return test_data_h2o


@custom
def ml_output(data, test_data):
    """
    This function trains an H2O AutoML model on the input data, uses the best model to make predictions on the test data, 
    and returns a pandas dataframe with the original test data and the predictions.

    Args:
        data (pandas.DataFrame): The input dataframe with the training data and features.
        test_data (pandas.DataFrame): The input dataframe with the test data and features.

    Returns:
        A pandas dataframe with the original test data and the predictions appended as a new column.

    """
    
    leader_model = ml_model(data)
    preds_df = make_predictions(leader_model, test_data)

    return preds_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
