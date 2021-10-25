### remove missing etc
import os
import pandas as pd
import valohai

inputs = {"attacksmini":[]}
valohai.prepare(step="pre_process", default_inputs=inputs)

file_path = valohai.inputs('attacksmini').path()
attacks = pd.read_csv(file_path)

# remove where sharks were not involved
attacks=attacks[~attacks['Species'].str.contains('Questionable|Shark involvement not confirmed|Shark involvement prior|No shark involvement')]


# Set your model target as categorical
attacks['Species'] = pd.Categorical(attacks['Species'])

path = valohai.outputs('*').path('attacksminiprocessed.csv')
attacks.to_csv(path)
