if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import h2o
from h2o.automl import H2OAutoML
h2o.init()

def ml_model(df):
    df_h2o = h2o.H2OFrame(df)
    aml = H2OAutoML(max_runtime_secs = 60, seed = 1, nfolds = 10, project_name = "fds_fin_metric")
    aml.train(y = 'Stock_Percent_Change', training_frame = df_h2o)
    return aml.leaderboard.head()


@custom
def ml_output(data, test_data):
    """
    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    
    leaderboard = ml_model(data)

    print(leaderboard)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
