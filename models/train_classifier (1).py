import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download(['punkt','wordnet'])
import re
import pickle
from joblib import dump,load
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def load_data(database_filepath):
    # read in file
    engine = create_engine('sqlite:////home/workspace/data/DisasterResponse.db')
    # load to database
    df = pd.read_sql_table('TableETL',engine)
    # Add a check for extra columns
    expected_columns = ['id', 'message', 'original', 'genre', 'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']  # Replace with  actual column names
    actual_columns = df.columns.tolist()

    # Check for columns that are in actual_columns but not in expected_columns
    extra_columns = [col for col in actual_columns if col not in expected_columns]
    if extra_columns:
        print('Extra columns:', extra_columns)

    # Check for columns that are in expected_columns but not in actual_columns
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    if missing_columns:
        print('Missing columns:', missing_columns)
        
    # define features and label arrays
    X=df['message']
    y=df.drop(['id','message','original','genre'],axis=1) 
    # clean data
    # drop rows in y that contain '2'
#     clean = (y != 2).all(axis=1)
#     y = y[clean]
#     X = X[clean]
    category_names = Y.columns.tolist()
    return X, y,category_names

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    detected_urls=re.findall(url_regex,text)
    for url in detected_urls:
        text=text.replace(url,'urlplaceholder')
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    for tok in tokens:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def display_results(y_test,y_pred):
    for i, col in enumerate(y_test.columns):
        print(f"Column: {col}")
        labels = np.unique(y_pred)
        accuracy = (y_pred[:, i] == y_test[col]).mean()
        print(classification_report(y_test[col], y_pred[:, i]))
        print("Labels:", labels)
        print("Accuracy:", accuracy)
        print("\n")
 
def plot_scores(y_test, y_pred):
    f1_scores=[]
    labels=y_test.columns.tolist()
    for i,col in enumerate(y_test.columns):
        score=f1_score(y_test[col],y_pred[:,i],average='weighted')
        f1_scores.append(score)
        
    plt.figure(figsize=(10,5))
    plt.bar(labels,f1_scores)
    plt.xlabel('Classes')
    plt.ylabel('F1_Score')
    plt.title('F1-score for each class')
    plt.xticks(rotation=90)
    plt.show()
    
def build_model():
    X,y=load_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y)
    
    pipeline=Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    pipeline.fit(X_train,y_train)
    y_pred=pipeline.predict(X_test)
    display_results(y_test,y_pred)
    plot_scores(y_test, y_pred)

def evaluate_model(model, X_test, y_test, category_names):
     y_pred = model.predict(X_test)
     for i in range(len(category_names)):
        print("Category:",category_names[i],"\n",classification_report(y_test[:, i],y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


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