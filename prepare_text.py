##tokenize using transformers
import os
import pandas as pd
import valohai
from transformers import DistilBertConfig, DistilBertTokenizerFast
import torch
from sklearn.model_selection import train_test_split
import pickle

inputs = {"attacksminiprocessed":[]}
valohai.prepare(step="pepare_text", default_inputs=inputs)

file_path = valohai.inputs('attacksminiprocessed').path()
#attacks = pd.read_csv('attacksminiprocessed.csv')
attacks = pd.read_csv(file_path)
#attacks.size
mostcommon=attacks['Species'].value_counts().nlargest(20).to_frame('counts').reset_index().rename(columns={'index': 'Species'})
#mostcommon

#filter df to contain only the most common
attacks=attacks[attacks.Species.isin(mostcommon['Species'])]

# Transform your output to numeric
#attacks['Species_int'] = attacks['Species'].cat.codes
attacks['Species_int']=attacks['Species'].apply(lambda x:mostcommon["Species"].to_list().index(x))


df=attacks[['Injury','Species','Species_int']]

#check still that there are no missing values
df = df.dropna()

# Train set, validation set and test set
from sklearn.model_selection import train_test_split

train_val, test = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
train, val = train_test_split(train_val, test_size=0.2, random_state=42, shuffle=True)

# Reset the indexes of the 3 pandas.DataFrame()
train, val, test = map(lambda x:x.reset_index(drop=True), [train, val, test])


#save dict for further use
out_path = valohai.outputs().path('my_dict.csv')
mostcommon.to_csv(out_path)

out_path = valohai.outputs().path('train.csv')
train.to_csv(out_path)

out_path = valohai.outputs().path('test.csv')
test.to_csv(out_path)

out_path = valohai.outputs().path('val.csv')
val.to_csv(out_path)
