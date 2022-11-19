import sys

# download necessary NLTK data
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download(['punkt', 'wordnet'])
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    load data from sqlite file path. which is provided as parameter.
    Parameters:
    * database_filepath: sqlite db file path
    '''
    # load data from database
    engine = create_engine('sqlite:///%s' % database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df.message
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X.values, y.values, category_names


def tokenize(text):
    '''
    create token from input text.
    Parameters:
    * text: input text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    build model to prepare for training
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        # comment for return quick response.
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=2, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaludate model and printing classify report.
    '''
# below comment codes are for testing purposes
#     Y_pred = model.predict(X_test)
#     labels = np.unique(Y_pred)
#     confusion_mat = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1), labels=labels)
#     accuracy = (Y_pred == Y_test).mean()

#     print("Labels:", labels)
#     print("Confusion Matrix:\n", confusion_mat)
#     print("Accuracy:", accuracy)
#     print("\nBest Parameters:", model.best_params_)
    Y_pred = model.predict(X_test)
    evaluate_report = classification_report(Y_test, Y_pred, target_names=category_names)
    print(evaluate_report)


def save_model(model, model_filepath):
    '''
    save model to file which is provided file path
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