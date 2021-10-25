##tokenize using transformers
import os
import pandas as pd
import valohai
from transformers import DistilBertConfig, DistilBertTokenizerFast
import torch
from sklearn.model_selection import train_test_split
import pickle

inputs = {"attacksminiprocessed":[]}
valohai.prepare(step="tokenize_text", default_inputs=inputs)

file_path = valohai.inputs('attacksminiprocessed').path()
#attacks = pd.read_csv('attacksminiprocessed.csv')
attacks = pd.read_csv(file_path)

attacks['Species'] = pd.Categorical(attacks['Species'])
mostcommon=attacks['Species'].value_counts().nlargest(40).to_frame('counts').reset_index()
#mostcommon
attacks=attacks[attacks.Species.isin(mostcommon['index'])]
#attacks.head(50)
# Transform your output to numeric
attacks['Species_int'] = attacks['Species'].cat.codes
my_dict=attacks[['Species','Species_int']].to_dict('dict')


df=attacks[['Injury','Species','Species_int']]

#check still that there are no missing values
df = df.dropna()

# Train set, validation set and test set
from sklearn.model_selection import train_test_split

train_val, test = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
train, val = train_test_split(train_val, test_size=0.2, random_state=42, shuffle=True)

# Reset the indexes of the 3 pandas.DataFrame()
train, val, test = map(lambda x:x.reset_index(drop=True), [train, val, test])

# Load Distilbert's tokenizer
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenizer.max_model_input_sizes

# Format the train/validation/test sets
train_encodings = tokenizer(train['Injury'].to_list(), truncation=True, padding=True)
val_encodings = tokenizer(val['Injury'].to_list(), truncation=True, padding=True)
test_encodings = tokenizer(test['Injury'].to_list(), truncation=True, padding=True)


#save dict for further use
with open('my_dict.dictionary', 'wb') as config_dictionary_file:
    pickle.dump(my_dict, config_dictionary_file)

#save data
with open('train_encodings', 'wb') as train_encodings_file:
    pickle.dump(train_encodings, train_encodings_file)

with open('val_encodings', 'wb') as val_encodings_file:
    pickle.dump(val_encodings, val_encodings_file)

with open('test_encodings', 'wb') as test_encodings_file:
    pickle.dump(test_encodings, test_encodings_file)



out_path = valohai.outputs().path('train.csv')
train.to_csv(out_path)

out_path = valohai.outputs().path('test.csv')
test.to_csv(out_path)

out_path = valohai.outputs().path('val.csv')
val.to_csv(out_path)
