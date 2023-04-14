if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()

# Define a function to compute sentiment scores for a list of texts
def compute_sentiment_scores(text):
    """
    Computes the average compound sentiment score for a list of texts.
    """
    score = analyzer.polarity_scores(text)
    return score['compound']

@transformer
def transform(df, *args, **kwargs):
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


    df['headline_sentiment_score'] = df['headline'].apply(compute_sentiment_scores)
    df['summary_sentiment_score'] = df['summary'].apply(compute_sentiment_scores)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
