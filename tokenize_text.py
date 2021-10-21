##tokenize using transformers
import os
import pandas as pd
import valohai
from transformers import DistilBertConfig, DistilBertTokenizerFast
import torch
from sklearn.model_selection import train_test_split

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
#my_dict=attacks[['Species','Species2']].to_dict('dict')

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


class my_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = my_Dataset(train_encodings, train['Species_int'].to_list())
val_dataset = my_Dataset(val_encodings, val['Species_int'].to_list())
test_dataset = my_Dataset(test_encodings, test['Species_int'].to_list())


#save dict for further use
output_path = 'my_dict'
model.save(output_path)

#save data
path = valohai.outputs('train_dataset').path('train_dataset')
train_dataset.save(path)

path = valohai.outputs('val_dataset').path('val_dataset')
val_dataset.save(path)

path = valohai.outputs('test_dataset').path('test_dataset')
test_dataset.save(path)
