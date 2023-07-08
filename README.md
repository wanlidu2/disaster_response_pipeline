# disaster_response_pipeline
# import packages
import sys
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt','wordnet'])
import re
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

def load_data(data_file):
    # read in file
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    # load to database
    df = pd.read_sql_table('TableETL',engine)
    # define features and label arrays
    X=df['message']
    y=df.drop(['id','message','original','genre'],axis=1) 
    # clean data
    # drop rows in y that contain '2'
    clean = (y != 2).all(axis=1)
    y = y[clean]
    X = X[clean]
    return X, y

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

def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)

def main():
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

##check whether drop y=2
    unique_values = np.unique(y)
    print(unique_values)
    count = (y == 2).sum().sum()
    print(f"Number of times 2 appears in y: {count}")

main()
