if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def lowercase_text(df):
    df['headline'] = df['headline'].str.lower()
    df['summary'] = df['summary'].str.lower()
    return df

def remove_special_char(df):
    if df is None:
        print("Error: DataFrame is None.")
        return None
    elif 'headline' not in df.columns or 'summary' not in df.columns:
        print("Error: 'headline' or 'summary' columns not found in DataFrame.")
        return None
    else:
        # Remove special characters
        df['headline'] = df['headline'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
        df['summary'] = df['summary'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
        return df

def remove_tags(df):
    # Remove HTML tags and URLs (if applicable)
    df['headline'] = df['headline'].replace(r'<.*?>', '', regex=True)
    df['summary'] = df['summary'].replace(r'<.*?>', '', regex=True)
    df['headline'] = df['headline'].replace(r'http\S+', '', regex=True)
    df['summary'] = df['summary'].replace(r'http\S+', '', regex=True)
    return df

def remove_stopwords(df):
    # Remove stop words
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    df['headline'] = df['headline'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
    df['summary'] = df['summary'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
    return df

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
    df = lowercase_text(df)

    df = remove_special_char(df)

    df = remove_tags(df)

    df = remove_stopwords(df)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
