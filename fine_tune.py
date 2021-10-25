
import valohai
import pickle
import torch
import pandas as pd

# inputs
inputs = {"train_dataset":[],"val_dataset":[],"test_dataset":[]}
valohai.prepare(step="fine_tune", default_inputs=inputs)

file_to_read = open("train_encodings", "rb")
train_encodings = pickle.load(file_to_read)

file_to_read = open("val_encodings", "rb")
val_encodings = pickle.load(file_to_read)

file_to_read = open("test_encodings", "rb")
test_encodings = pickle.load(file_to_read)

file_path = valohai.inputs('train').path()
train = pd.read_csv(file_path)

file_path = valohai.inputs('test').path()
test = pd.read_csv(file_path)

file_path = valohai.inputs('val').path()
val = pd.read_csv(file_path)

# make torch dataset

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


# Fine-tuning with Trainer
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
)

# The number of predicted labels must be specified with num_labels
# .to('cuda') to do the training on the GPU
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(df['Species'].to_list())).to('cuda')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model("shark_model")


# Accuracy metrics
predictions=trainer.predict(test_dataset)

test_results=test.copy(deep=True)
test_results["label_int_pred_transfer_learning"]=predictions.label_ids
test_results['label_pred_transfer_learning']=test_results['label_int_pred_transfer_learning'].apply(lambda x:labels[x])

error_rate_sum_zs=len(test_results[test_results["label"]!=test_results["label_pred_transfer_learning"]])/len(test_results)

# Print out metrics
with valohai.metadata.logger() as logger:
    logger.log("accuracy", error_rate_sum_zs)
