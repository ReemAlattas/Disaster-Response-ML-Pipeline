import json
import plotly
import pandas as pd

import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('LabeledMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    related_counts = df.related.value_counts()
    related_names = ['related', 'not related']
    
    request_counts = list(df[df.columns[5:7]].sum())
    request_names = df.columns[5:7]
    
    category_df = df[df.columns[7:]].sum()
    category_counts = list(category_df.sort_values(ascending=False))
    category_names = df.columns[7:]
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Disaster Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=request_names,
                    y=request_counts
                )
            ],

            'layout': {
                'title': 'Request vs Offer Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Requests"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Features",
                    'tickangle':315
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
