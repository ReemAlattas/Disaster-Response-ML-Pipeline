import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    - Load the messages and categories datasets from csv files into dataframes
    - Merge the messages and categories datasets using the common id

            Parameters:
                    messages_filepath (str): Messages dataset file path
                    categories_filepath (str): Categories dataset file path

            Returns:
                    df (dataframe): Dataframe of combined datasets
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath, encoding='utf-8')
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath, encoding='utf-8')
    
    # merge datasets
    df = pd.merge(messages, categories)
    
    return df

def clean_data(df):
    '''
    Cleans the data:
       - Split the values in the categories column on the ; character so that each value becomes a separate column
       - Create column names for the categories data
       - Rename columns of categories with new column names
       - Convert category values to numbers 0 or 1
       - Drop duplicates

            Parameters:
                    df (dataframe): Dataframe of combined messages and categories

            Returns:
                    df (dataframe): Clean dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [(lambda r: r[:-2])(r) for r in row]
   
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    
    # replace 2 with 1 values
    df.replace(2, 1, inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Save the clean dataset into an sqlite database

            Parameters:
                    df (dataframe): Clean dataframe
                    database_filename (str): Sqlite database file path

    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('LabeledMessages', engine, index=False)


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
