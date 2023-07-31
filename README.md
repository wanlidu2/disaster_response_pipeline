
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
ETL stands for Extract, Transform, and Load process. In this part, I import the dataset, check the data structure, clean the data, and save the useful part in an SQLite database. Processing in ETL can ensure the data is understandable, clean, and usable. It allows more complicated transformation of data and improves performance at the same time.

The importance of using SQLite database is that SQLite has better data integrity and is more efficient in both data access and data storage.

The whole processing code was saved as the process_data.py.

-2 Machine Learning Pipeline

The common step before having machine learning algorithms is to split the data into two parts, training, and test set. Since in this task, we need to handle the text input, we can use NLTK, the Natural Language Toolkit, a library in Python that provides tools dealing with human language data. It contains libraries for classification, tokenization, stemming, and other process, for example, the WordNet. Besides, I also use scikit-learn's Pipeline and GridSearchCV for constructing my model of machine learning.

-3 Flask Web app

With the help of the basic command framework, I applied knowledge of Flask, html to finally display my work on the website. I modified the paths of the data in the SQLite database and my previous model was saved in Python notebook. Then I call Plotly in the web app.

## Strategy

![image](https://github.com/wanlidu2/disaster_response_pipeline/assets/121735612/965dca1e-736f-4662-9a3f-1d595da765f1)
code construction

-1 Stage 1 - ETL

Purpose: 

In the beginning, I was going to handle the data extraction, transformation, and loading to a more convenient place to store and read.

Code: 

Import and read the datasets;
Merge two datasets: messages.csv and categories.csv on the 'id' column;
Split the values from the 'categories' column by the sign ';' and reorganize the column, making it more clear to read;
Display the columns after splitting, using the lambda function to extract each string;
Get dummies from the category values to guarantee the information could be used in the pipeline and regression;
Drop the original categories from the dataset since we already store the more structured information in new columns, it is for the code to be more clean and the dataset more readable;
Drop the duplicated columns in case to simplify the code before regression and save time;
Using to_sql to save the clean dataset into an SQLite database for future use;

Results and effects:

To summarize, I examined the comprehensive state of the data, merging various datasets, and extracted actionable insights from raw messages, subsequently storing this information. This data processing stage allowed me to refine the data, making it clearer and more readily usable for future applications. This critical step served as the foundational groundwork for the subsequent stages and was thus indispensable.

-2 Stage 2 - ML Pipeline

Purpose:

Code:

Results and effects:

-3 Stage 3 - IDE

Purpose:

Code:

Results and effects:


## Challenges and some problem solutions


## Display of Results



## Reflection and Insights
