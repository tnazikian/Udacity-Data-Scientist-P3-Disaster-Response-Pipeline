import sys
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
import re
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, ngrams
from nltk import download
from sqlalchemy import create_engine
import pickle

def load_data(database_filepath):
    categories = ['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df['message'].values
    cat_values = df.loc[:,categories]
    Y = cat_values
    return X, Y, categories

def tokenize(text):
    # Make it lower
    text = text.lower().strip()
    lemmatizer = WordNetLemmatizer()

    # Define a pattern to filter non-alphanumerics
    clean_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    words = clean_text.split(' ')
    lem = [lemmatizer.lemmatize(word) for word in words]
    return lem

def tokenize_with_bigrams(text):
    # Tokenize the text into individual words
    tokens = tokenize(text)

    # Generate bigrams from the tokens
    bigrams = list(ngrams(tokens, 2))

    # Convert bigrams to strings
    bigrams = [' '.join(bigram) for bigram in bigrams]

    # Combine the original tokens and bigrams
    combined_tokens = tokens + bigrams

    return combined_tokens

def build_model():
    return Pipeline([
    ('FeatureUnion', FeatureUnion([
        ('etl_union', Pipeline([
            ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
            ('tfidf_transformer', TfidfTransformer())
        ])),
        ('bigram_pipe', Pipeline([
            ('bigram_tfidf', TfidfVectorizer(tokenizer=tokenize_with_bigrams))
        ]))
    ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

def evaluate_model(model, X_test, Y_test, category_names):
    y_prediction_test = model.predict(X_test)
    df_pred = pd.DataFrame(Y_test, columns=category_names)
    for col in category_names: 
        print(classification_report(Y_test[col], df_pred[col]))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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