# Disaster Response Pipeline Project
## Libraries used:
    libraries required for project are in requirements.txt file.

setup:
	# Create python virtualenv & source it
	pip3 install virtualenv
	python3.7.3 -m venv ~/.ds_project2 
	source ~/.ds_project2/bin/activate

install:
	# This should be run from inside a virtualenv
	pip install --upgrade pip --ignore-installed TBB &&\
		pip install -r requirements.txt
cleanenv:
	rm -rf ~/.ds_project2 
## Installation:
    - install libraries by command line: pip install -r requirements.txt
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
