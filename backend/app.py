import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'helpers/pitchesdeals.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    pitches_df = pd.DataFrame.from_dict(data, orient='index').reset_index(drop=True)

app = Flask(__name__)
CORS(app)


def json_search(query):

    query_words = set(query.lower().split())
    query_word_count = len(query_words)
    # check if any query word is in the pitch description
    def is_match(desc):
        desc_words = set(desc.lower().split())
        return not query_words.isdisjoint(desc_words)  # true if any word in common

    matches = pitches_df[pitches_df['Pitched_Business_Desc'].apply(is_match)]
    # matches = pitches_df[pitches_df['Pitched_Business_Desc'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['Pitched_Business_Identifier', 'Pitched_Business_Desc', 'Deal_Status', 'Deal_Shark', 'US_Viewership']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/pitches")
def pitches_search():
    # Get the user input from query parameters
    text = request.args.get("query")

    # Check if the text parameter is provided
    if not text:
        # Return an error message or an empty JSON if no query parameter is provided
        return {"error": "No query provided"}, 400  # 400 Bad Request

    # Call the search function with the user input
    result = json_search(text)

    # Check if the search result is empty
    if not result:
        # Return a message indicating no matches found
        return {"message": "No matches found"}, 404  # 404 Not Found

    return result


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
