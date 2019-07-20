# Disaster Response Pipeline Project

### Project Overview & Motivation
The project is a code that can be used to classify messages received in case of emergencies (e.g. flooding, severe weather, etc) into relevant messages and into different categories to be directed to the right agency. This is completed for project 6 of the Udacity Nanodegree for data science.

### Files
# process_data.py
This file contains the code responsible for implementing the ETL pipeline by merging the messages with the different categories, cleaning the data and storing it in a SQL database.

# train_classifier.py
This file is for the ML pipeline where we load the data from the SQL database and split it into trainnin and testing data, build, train and utilize maching learning pipeline to predict the categories of any message.

# run.py
This is the Flask web app file responsible for running the server and utilize the previous work to vizualize and provide a convinent way of predicting the categories for new messages.

# .csv
These are the originally provided datasets

# DisasterResponse.db
This is the output from the ETL pipeline stored as a SQLite database

### Libraries
The following libraries are utilized:
# Scikit-Learn, Pandas, NumPy
# NLTK
# SQLalchemy, Flask, Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

