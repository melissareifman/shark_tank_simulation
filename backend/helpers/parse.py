"""
import pandas

import json

excel_data_df = pandas.read_excel('Sharktankpitchesdeals.xlsx', sheet_name='Sharktankpitchesdeals')

json_str = excel_data_df.to_json()

with open('pitchesdeals.json', 'w') as f:
    json.dump(json_str, f)
"""

import csv
import json
 
def make_json(csvFilePath, jsonFilePath):
    data = {}
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            print(rows)
            key = rows['Pitched_Business_Identifier']
            data[key] = rows
 
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))


csvFilePath = './parse.py'
jsonFilePath = './pitchesdeals.json'
 
make_json(csvFilePath, jsonFilePath)