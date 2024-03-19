import pandas

import json

excel_data_df = pandas.read_excel('Sharktankpitchesdeals.xlsx', sheet_name='Sharktankpitchesdeals')

json_str = excel_data_df.to_json()

with open('pitchesdeals.json', 'w') as f:
    json.dump(json_str, f)
