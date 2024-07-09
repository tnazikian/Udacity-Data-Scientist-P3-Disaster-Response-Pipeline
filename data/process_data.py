import sys
import pandas as pd
from sqlalchemy import create_engine

"""
load_data
Loads disaster message data from csv files provided by Figure8 and 
stores in pandas dataframe.

Input:
messages_filepath    File path for disaster_message.csv file
categories_filepath  File path for disaster_categories.csv file

Output:
df    Pandas df containing merged data from both the message and categories files
"""
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

"""
clean_data
Cleans merged dataframe from the csv files

Input:
df    Pandas Dataframe containing merged data from both csv files

Output:
df    Dataframe with cleaned/new data
"""
def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row]
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop any rows where the category value is not binary (0, 1)
    for column in categories:
        df = df[df[column].isin([0, 1])]
    
    return df

"""
save_data
Saves a dataframe to the specified target filename as a sqlite db file

Input:
df    Pandas Dataframe containing merged/cleaned data
"""
def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()