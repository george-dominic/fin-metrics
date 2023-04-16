if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


def lowercase_text(df: pd.DataFrame) -> pd.DataFrame:
    """
        Function to convert text data in a DataFrame to lowercase.

        Parameters:
        -----------
        df : pandas DataFrame
            Input DataFrame containing text data.

        Returns:
        --------
        df : pandas DataFrame
            DataFrame with text data converted to lowercase.
    """
    df['headline'] = df['headline'].str.lower()
    df['summary'] = df['summary'].str.lower()
    return df

def remove_special_char(df: pd.DataFrame) -> pd.DataFrame:
    """
        This function takes a DataFrame 'df' as input and removes special characters from the 'headline' and 'summary' columns.
        It uses the 'apply' method along with a lambda function to replace any character that is not an uppercase or lowercase alphabet or a digit with a space.

        :param df: pd.DataFrame: Input DataFrame with 'headline' and 'summary' columns
        :return: pd.DataFrame: DataFrame with special characters removed from 'headline' and 'summary' columns
    """
    df['headline'] = df['headline'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
    df['summary'] = df['summary'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
    return df

def remove_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
       This function takes a DataFrame 'df' as input and removes HTML tags and URLs (if applicable) from the 'headline' and 'summary' columns.
       It uses the 'replace' method along with regular expressions to replace any HTML tag (denoted by '<' and '>' characters) with an empty string, and any URL (starting with 'http' and followed by any non-whitespace characters) with an empty string.

       :param df: pd.DataFrame: Input DataFrame with 'headline' and 'summary' columns
       :return: pd.DataFrame: DataFrame with HTML tags and URLs removed from 'headline' and 'summary' columns
    """
    df['headline'] = df['headline'].replace(r'<.*?>', '', regex=True)
    df['summary'] = df['summary'].replace(r'<.*?>', '', regex=True)
    df['headline'] = df['headline'].replace(r'http\S+', '', regex=True)
    df['summary'] = df['summary'].replace(r'http\S+', '', regex=True)
    return df

def remove_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame 'df' as input and removes stop words from the 'headline' and 'summary' columns.
    It first downloads the necessary NLTK resources for stop words, tokenization, and lemmatization.
    Then, it uses the 'stopwords' module from NLTK to obtain a set of stop words in English.
    Next, it uses the 'apply' method along with a lambda function to tokenize the text in 'headline' and 'summary' columns using 'word_tokenize' from NLTK.
    It further filters out the stop words from the tokenized text using a list comprehension.
    Finally, it joins the filtered tokens back into a string using the 'join' method.

    :param df: pd.DataFrame: Input DataFrame with 'headline' and 'summary' columns
    :return: pd.DataFrame: DataFrame with stop words removed from 'headline' and 'summary' columns
    """

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    df['headline'] = df['headline'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
    df['summary'] = df['summary'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
    return df

@transformer
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
        This function takes a DataFrame 'df' as input and applies a series of transformations on it.
        It calls four helper functions, 'lowercase_text', 'remove_special_char', 'remove_tags', and 'remove_stopwords', sequentially to perform the transformations.
        The 'lowercase_text' function converts all text in the DataFrame to lowercase.
        The 'remove_special_char' function removes special characters from the 'headline' and 'summary' columns.
        The 'remove_tags' function removes HTML tags and URLs (if applicable) from the 'headline' and 'summary' columns.
        The 'remove_stopwords' function removes stop words from the 'headline' and 'summary' columns using NLTK.
        The transformed DataFrame is returned at the end.

        :param df: pd.DataFrame: Input DataFrame
        :return: pd.DataFrame: Transformed DataFrame
    """
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
