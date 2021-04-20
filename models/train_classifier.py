import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    - Load dataset from sqlite database
    - Define feature and target variables X and Y

            Parameters:
                    database_filepath (str): Sqlite database file path

            Returns:
                    X (dataframe): Feature variable
                    Y (dataframe): Target variable
                    Y.columns (list of str): Target column names
    ''' 

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM LabeledMessages", engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    tokenization function to process disaster messages

            Parameters:
                    text (str): A disaster message

            Returns:
                    tokens (list of str): A list of normalized and lemmatized tokens
    '''
    # Normalize text  
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text 
    tokens = word_tokenize(text) 
    
    # Remove stop words 
    tokens = [t for t in tokens if t not in stopwords.words("english")] 
    
    # Lemmatization - Reduce words to their root form 
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    
    # Stemming - Reduce words to their stems
    tokens = [PorterStemmer().stem(t) for t in tokens] 
     
    return tokens


def build_model():
    '''
    Build machine learning pipeline

            Returns:
                    cv (class):  ML pipeline for predicting multiple target variables
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(model, param_grid=parameters)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Report the f1 score, precision and recall for each output category of the dataset

            Parameters:
                    model (class): ML pipeline for predicting multiple target variables
                    X_test (str): Feature variable test set
                    Y_test (str): Target variable test set
                    category_names (list of str): Column names of target variable test set
    '''
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Exports the model as a pickle file

            Parameters:
                    model (class): ML pipeline for predicting multiple target variables
                    model_filepath (str): Output pickle file path
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
