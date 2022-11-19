# Disaster Response Pipeline Project
## Libraries used:
    python 3.7.9 + libraries required for project are in requirements.txt file.

## Installation:
    - install libraries by command line: pip install -r requirements.txt
	
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run your web app: `python app/run.py`

## Project summary
    - Project name: Disaster Response Project
    - Description: Project is to predict real message to 36 classified categories.

## explanation of files in the repository
    - app folder: contains source code of web app
		+ templates folder: contains html template
		+ run.py: flask python web app source code
    - data: contains data input, output and source code to clean data
		+ disaster_categories.csv: input data of disaster categories
		+ disaster_messages.csv: input disaster messages data
		+ DisasterResponse.db: sqlite cleaned database exported when run source code file process_data.py
		+ process_data.py: source code file to load input data, clean data, and export data to sqlite database file.
    - models folder contains classified model and train_classifier file code.
		+ classifier.pkl: file model exported from train_classifier.py file code
		+ train_classifier.py: source code file to export classifier.pkl model
    - requirements.txt contains all libraries required to work with this project.

## Licensing, Author, and Acknowledgements
    - Acknowledgements: disaster data files input from Appen website (https://www.figure-eight.com/)
    - Author: MinhNT34
    - License: Distributed under the MIT License.