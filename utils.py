import re
import openai
import tiktoken
import pandas as pd
from scipy import spatial
from collections import namedtuple
from dateutil.parser import parse

def set_api_key(key, url):
    """
    Function takes a string which is either a key
    """
    openai.api_key = key
    openai.api_base = url


def create_config(content):
    """
    This function takes a dictionary and returns a named tuple.
    """
    config = namedtuple(typename='Config', field_names=content.keys())
    return config._make(content.values())


def clean_web_data(df):
    """
    This functions takes a Pandas dataframe with column 'text' and does
    some cleaning and preprocessing on this column:
      - removing all rows after the heading '== See also =='
      - removing empty rows
      - removing headings
      - removing unicode characters
      - removing inverted commas
      - exploding mulit-sentence rows into separate rows
    The result is returned as a copy of the original dataframe.
    """
    df_copy = df.copy()
    
    # Remove all rows after "== See also =="
    df_copy = df_copy[: df_copy[df_copy['text'] == '== See also =='].index[0]]
    
    # Clean up text to remove empty lines and headings
    df_copy = df_copy[(df_copy['text'].str.len() > 0) & (~df_copy['text'].str.startswith('=='))]
    
    # Remove unicode characters from column text
    df_copy['text'] = df_copy['text'].apply(lambda s: s.encode('utf-8', 'replace').decode())
    
    # Remove inverted commas
    df_copy['text'] = df_copy['text'].apply(lambda s: s.replace('"',''))
    
    # Split ever multi-sentence row into seperate rows
    df_copy['text'] = df_copy['text'].apply(lambda s: re.split(r'(?<=[\.\!\?])\s*', s))
    df_copy = df_copy.explode('text').reset_index(drop=True)
                                       
    # Remove empty lines created by the preprocessing
    df_copy = df_copy[df_copy['text'].str.len() > 0]

    return df_copy


def clean_csv_data(df):
    """
    Cleans a fashion trends CSV dataframe by:
      - Removing rows with missing or empty 'Trends' or 'Source'
      - Combining 'Trends' and 'Source' columns into a single 'text' column
      - Removing extra whitespace and quotes
      - Exploding multi-sentence rows into separate rows
    Returns a cleaned dataframe with a 'text' column.
    """
    
    df_copy = df.copy()
    
    # Remove rows with missing or empty 'Trends' or 'Source'
    df_copy = df_copy[df_copy['Trends'].notnull() & df_copy['Source'].notnull()]
    df_copy = df_copy[(df_copy['Trends'].str.strip() != '') & (df_copy['Source'].str.strip() != '')]

    # Combine 'Trends' and 'Source' into 'text'
    df_copy['text'] = df_copy['Trends'].astype(str).str.strip() + ' ' + df_copy['Source'].astype(str).str.strip()

    # Remove quotes and extra whitespace
    df_copy['text'] = df_copy['text'].str.replace('"', '', regex=False).str.replace("'", '', regex=False)
    df_copy['text'] = df_copy['text'].str.replace('  ', ' ', regex=False)
    df_copy['text'] = df_copy['text'].str.strip()

    # Explode multi-sentence rows into separate rows
    df_copy['text'] = df_copy['text'].apply(lambda s: re.split(r'(?<=[\.!?])\s*', s))
    df_copy = df_copy.explode('text').reset_index(drop=True)
    df_copy = df_copy[df_copy['text'].str.len() > 0]
    
    # Remove duplicated rows
    df_copy = df_copy.drop_duplicates(subset=['text']).reset_index(drop=True)
    return df_copy[['text']]


def get_embeddings(df, config, text_col='text', embedding_col='embedding'):
    """
    This function takes a dataframe, a configuration object where each entry
    can be access by name (dot notation), and the name of the text column
    in the dataframe and the name of the embedding column of the data returned
    by OpenAI. It returns a list of embeddings for each text (row) of the 
    dataframe.
    """
    response = openai.embeddings.create(
        input=df[text_col].tolist(),
        model=config.EMBEDDING_MODEL_NAME
    )
    return [data.embedding for data in response.data]

def distances_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
    distance_metric="cosine",
) -> list[list]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def get_embedding(text, model):
    """
    Get the embedding for a single text string using OpenAI's API (openai>=1.0.0).
    """
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def get_rows_sorted_by_relevance(question, df, config):
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns a copy of that
    dataframe sorted from least to most relevant for that question
    """
    # Get embedding for the question text
    question_embedding = get_embedding(question, config.EMBEDDING_MODEL_NAME)

    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(
        question_embedding,
        df_copy["embeddings"].values,
        distance_metric="cosine"
    )

    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy


def create_prompt(question, df, config, custom):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding(config.ENCODING)

    # Count the number of tokens in the prompt template and question
    prompt_template = """
        Answer the question based on the context below, and if the question
        can't be answered based on the context, say "I don't know"

        Context: 

        {}

        ---

        Question: {}
        Answer:
    """

    current_token_count = len(tokenizer.encode(prompt_template)) + \
                            len(tokenizer.encode(question))

    context = []
    if custom == True:
        for text in get_rows_sorted_by_relevance(question, df, config)["text"].values:

            # Increase the counter based on the number of tokens in this row
            text_token_count = len(tokenizer.encode(text))
            current_token_count += text_token_count

            # Add the row of text to the list if we haven't exceeded the max
            if current_token_count <= config.MAX_PROMPT_TOKENS:
                context.append(text)
            else:
                break

    return prompt_template.format("\n\n###\n\n".join(context), question)


def answer_question(question, df, config, custom=True):
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response - both values
    are contained in the config object, return the answer to the question 
    according to an OpenAI Completion model specified in config.

    If the model produces an error, return an empty string.
    """
    prompt = create_prompt(question, df, config, custom)

    try:
        response = openai.chat.completions.create(
            model=config.COMPLETION_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.MAX_RESPONSE_TOKENS
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(e)
        return ""
