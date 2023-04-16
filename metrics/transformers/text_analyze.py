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
      Transforms the input data frame by adding sentiment scores columns to the 'headline' and 'summary' columns.

      This function applies the 'compute_sentiment_scores' function to the 'headline' and 'summary' columns of
      the input data frame, calculating sentiment scores for each text using a sentiment analysis algorithm.
      The resulting sentiment scores are added as new columns to the input data frame, named 'headline_sentiment_score'
      and 'summary_sentiment_score' respectively.

      Args:
          df (pd.DataFrame): The input data frame from upstream blocks
          *args: Additional arguments, if any
          **kwargs: Additional keyword arguments, if any

      Returns:
          pd.DataFrame: Transformed data frame with added sentiment score columns
    """

    df['headline_sentiment_score'] = df['headline'].apply(compute_sentiment_scores)
    df['summary_sentiment_score'] = df['summary'].apply(compute_sentiment_scores)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
