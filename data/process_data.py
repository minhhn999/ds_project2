import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function is to load data into dataframe from data path files as parameter
    Parameters: 
    * messages_filepath: path file of messages data
    * categories_filepath: path file of categories data
    
    Returns: dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    '''
    This function is to clean dataframe with input as dataframe.
    Clean data include split categories column, remove duplicated, remove NaN data which is not valued for prediction.
    Parameters: 
    * df: input dataframe
    
    Returns: cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories.
    category_colnames = [(lambda x: x[0:-2])(x) for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # cleaning 'related' column
    categories.loc[(categories.related==2),'related']=0
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    # drop duplicated rows
    df.drop_duplicates(inplace=True)
    # drop rows contain Nan value
    df.dropna(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    save data from dataframe to sqlite database which is proviđe as parameter
    Parameters:
    * df: dataframe contains data to save
    * database_filename: filename path of sql database.
    '''
    engine = create_engine('sqlite:///%s' % database_filename)
    df.to_sql('InsertTableName', engine,if_exists = 'replace', index=False)
    # check data inserted.
    data = pd.read_sql('SELECT count(*) FROM InsertTableName', engine)
    print(data)


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