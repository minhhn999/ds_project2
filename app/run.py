import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import os
current_dir =  os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../")
print(parent_dir)

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
db_path = parent_dir + '/data/DisasterResponse.db'
print(db_path)
engine = create_engine('sqlite:///' + db_path)
df = pd.read_sql_table('InsertTableName', engine)

# load model
model_path = parent_dir + '/models/classifier.pkl'
model = joblib.load(model_path)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genres = df['genre'].value_counts()
    genre_names = list(genres.keys())
    genre_counts = list(genres.values)
    print(genres.value_counts())
    print(genre_names)
    print(genre_counts)
    categories = df.iloc[:, 4:]
    categories_names = categories.columns.tolist()
    categories_sum = categories[categories_names].sum().sort_values(ascending=False)
    categories_sum_names = categories_sum.index
    categories_sum_values = categories_sum.values
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_sum_names,
                    y=categories_sum_values
                )
            ],

            'layout': {
                'title': 'Distribution Total of Categories Matching',
                'yaxis': {
                    'title': "Total"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 30
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()