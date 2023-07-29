
# Disaster Response Pipeline

This is a data engineering project that mainly deals with data containing disaster information and builds a model to classify useful information for further analysis.

The final web app will be like 

![image](https://github.com/wanlidu2/disaster_response_pipeline/assets/121735612/e1d5894e-5f0f-498e-a3e6-400d15433557)

Given real data, we could have it through text analysis and get to know which kind of supplies is needed and give more direct and efficient help.

![image](https://github.com/wanlidu2/disaster_response_pipeline/assets/121735612/ba872887-fa22-4d2b-bb90-83fa5d2afd55)

## Task Allocation

I divided the projects into three main parts: the ETL Pipeline in Python, ML Pipeline also written in python jupyter notebook, Flask Web app.

## Project Details

-1 ETL Pipeline
ETL stands for Extract, Transform, and Load process. In this part, I import the dataset, check the data structure, clean the data, and save the useful part in a SQLite database. Processing in ETL can ensure the data is understandable, clean, and usable. It allows more complicated transformation of data and improves performance at the same time.

The importance of using SQLite database is that SQLite has better data integrity and is more efficient in both data access and data storage.

The whole processing code was saved as the process_data.py.

-2 Machine Learning Pipeline

The common step before having machine learning algorithems is split the data into two parts, training and test set. Since in this task, we need to handle the text input, we can use NLTK, the Natural Language Toolkit, a library in Python that provides tools dealing with human language data. It contains libraries for classification, tokenization, stemming and other process ,for example, the WordNet. Besides, I also use scikit-learn's Pipeline and GridSearchCV for constructing my model of machine learning.
