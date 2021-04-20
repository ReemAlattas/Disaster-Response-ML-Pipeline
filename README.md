# 1. Project Motivation
This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages.
The data set used in this project contains real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that the messages can be sent to the appropriate disaster relief agency.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app displays visualizations of the data as well. 

# 2. Installation
We use pandas to do the data cleaning.
To load the data into an SQLite database, we use an SQLAlchemy engine.
The ML pipeline uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output the final model.

# 3. File Descriptions
This repo contains Jupyter notebooks and Python scripts. Those files are described below:

## 3.1. Jupyter Notebooks
The Jupyter notebooks include instructions to get started with data pipelines: ETL and ML. The Jupyter notebooks were completed before working on the Python script.

### 3.1.1. Project Workspace - ETL
The first part of the data pipeline is the Extract, Transform, and Load process. In this notebook, we read the dataset, clean the data, and then store it in a SQLite database. 

### 3.1.2. Project Workspace - Machine Learning Pipeline
For the machine learning portion, we split the data into a training set and a test set. Then, we ceate a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The final model is exported to a pickle file.

## 3.2. Python Script Files
The python script files are included in the data and models folders.

### 3.2.1. process_data.py
This file has the final ETL script that extracts, transforms, and load the data into an SQLite database.

### 3.2.2. train_classifier.py
This file contains the final machine learning code that uses the message column to predict classifications for 36 categories.

## 3.3. Flask App
The results are diplayed in a Flask web app. The homepage contains 4 visualizations:
1. Disaster related messages vs non-related messages.
2. Request vs offer messages.
3. Distribution of message categories.
4. Distribution of message genres.

The emergency worker can use the web app to input a new message and get classification results in multiple categories.

# 4. How to Run the Web App
In a terminal window, follow these instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

The web app should now be running.

# 5. Acknowledgements 
Data provider: Figure Eight
