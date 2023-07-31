
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

To summarize, I examined the comprehensive state of the data, merging various datasets, and extracting actionable insights from raw messages, subsequently storing this information. This data processing stage allowed me to refine the data, making it clearer and more readily usable for future applications. This critical step served as the foundational groundwork for the subsequent stages and was thus indispensable.

-2 Stage 2 - ML Pipeline

Purpose:

Splitting the data into train and test datasets. After the text recognization and categorization processing, I construct a machine learning pipeline and then used GridSearchCV to improve the parameters setting and the accuracy of the prediction. The model was used to identify the 'message' input column and then predict classifications for 36 categories given by the previous datasets. At last, I saved my model into a pickle file for further use in the running process in the IDE part.

Code:

Import libraries: sys and pandas to read and process data, matplotlib for data visualization, pickle to save the model, nltk for text analysis, sklearn to implement the code of machining learning building;
The first was to load data from the engine I created by SQLite;
Then I wrote a tokenization function to process the text data;
Split the data;
Defining a main() function and constructing a Pipeline with CountVectorizer, TfidfTransformer, and MultiOutputClassifier;
Define display_result() function to show the scores of true values and the predicting accuracy;
Define plot_scores() to help evaluate the model's performance;
Using the GridSearchCV() to find out fitted parameters and improve the model;
Using the improved new model to test the data and generate better predictions;
Try other machine learning strategies such as svm to improve the model further;
Export the model and save it into a pickle file;

Results and effects:

I built a complete model and saved it in a pickle file, which could be invoked in the further model. After breaking down the task into machine learning parts, I could improve each small step more efficiently.

-3 Stage 3 - IDE in Linux

Purpose:

To display the model in a Flask web app. Primarily, I wrote the operations to call my model and the file storage locations into the code and ultimately presented the results in a visualized webpage

Code:
Run the 
ETL pipeline and saved the cleaned dataset in the database;
Run the ML pipeline and train the model then save the model in a pickle file;
Add the data and file path to the run.py then try to run the websites;

Results and effects:

I created an interactive webpage that allows for text input the analyzes it into 36 'message' columns.


## Challenges and some problem solutions

It was my first time using Linux and I could not find where to run the code of the file. Then I searched for instructions on the website and asked my mentor for help then get to know more useful skills to design the model and the interactive website. Besides, saving data in SQLite is also an efficient way to save data, and I will use it more frequently in further study.

## Display of Results

https://tli27l4ijp.prod.udacity-student-workspaces.com/


## Reflection and Insights

https://learn.udacity.com/nanodegrees/nd025/parts/cd0018/lessons/c5de7207-8fdb-4cd1-b700-b2d7ce292c26/concepts/068e995d-9064-4098-a0ce-e74dc7f90fa6
