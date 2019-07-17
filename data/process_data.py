import sys
import pandas as pd
import numpy as np
import sqlalchemy
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """ A function that loads the messages data and the categories data and merge them
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df


def clean_data(df):
    """ A function that cleans the data
    """
    # split from all values in one string to each value in a different column
    categories = df.categories.str.split(';', expand= True)
    
    # Change column names
    row = categories.loc[0,:]
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string to get the numeric
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace= True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis= 1)
    
    # drop duplicates
    df.drop_duplicates(inplace= True)
    
    return df


def save_data(df, database_filename):
    """ A function that saves the data to sql database
    """
#     conn = sqlite3.connect(database_filename)
#     df.to_sql('messages',conn)
#     conn.close


    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
#     print('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists = 'append')
    


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