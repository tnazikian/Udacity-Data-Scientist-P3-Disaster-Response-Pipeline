# Disaster Response Pipeline Project

### Summary:
This project creates an ETL (Extract, Transform, Load) and Machine Learning pipeline that categorizes various text messages that were sent during disasters. These messages are comprised of 36 pre-defined labels that define the type of disaster. The aim of this project is to automatically identify what kind of disaster is happening so that the right messages are sent to the appropriate disaster relief agency.

### Task Details:
This project uses a real-world dataset from Figure Eight containing real messages sent during disasters. Because each message can belong to multiple categories of disaster, making this a multi-label classification task. 

### File Description
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- README
~~~~~~~

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
