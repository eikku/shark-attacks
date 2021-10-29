
import valohai
import pickle
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast

# inputs
inputs = {"train":["train.csv"],"val":["val.csv"],"test":["test.csv"],"my_dict":["my_dict.csv"]}
valohai.prepare(step="fine_tune", default_inputs=inputs)

file_path = valohai.inputs('train').path()
train = pd.read_csv(file_path)

file_path = valohai.inputs('test').path()
test = pd.read_csv(file_path)

file_path = valohai.inputs('val').path()
val = pd.read_csv(file_path)

file_path = valohai.inputs('my_dict').path()
my_dict = pd.read_csv(file_path)

# Load Distilbert's tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenizer.max_model_input_sizes

# Format the train/validation/test sets
train_encodings = tokenizer(train['Injury'].to_list(), truncation=True, padding = "max_length")
val_encodings = tokenizer(val['Injury'].to_list(), truncation=True, padding = "max_length")
test_encodings = tokenizer(test['Injury'].to_list(), truncation=True, padding = "max_length")


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
    eval_steps=10,                      #print every 10 instead of default 500
    logging_steps=10,
)
# The number of predicted labels must be specified with num_labels
# .to('cuda') to do the training on the GPU
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(my_dict['Species'].to_list())).to('cuda')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

trainer.state.log_history

#print evaluation metrics
metrics=trainer.evaluate()
print(metrics)

out_path = valohai.outputs().path('shark_model')
trainer.save_model(out_path)
tokenizer.save_pretrained(out_path)

# Accuracy metrics
predictions=trainer.predict(test_dataset)

test_results=test.copy(deep=True)
test_results["label_int_pred_transfer_learning"]=predictions.label_ids
test_results['label_pred_transfer_learning']=test_results['label_int_pred_transfer_learning'].apply(lambda x:my_dict["Species"][x])

error_rate_sum_zs=len(test_results[test_results["Species"]!=test_results["label_pred_transfer_learning"]])/len(test_results)

# Print out metrics
with valohai.metadata.logger() as logger:
    logger.log("error_rate", error_rate_sum_zs)
    logger.log("eval_loss", metrics['eval_loss'])
    logger.log("eval_runtime", metrics['eval_runtime'])
    logger.log("eval_samples_per_second", metrics['eval_samples_per_second'])
    logger.log("eval_steps_per_second", metrics['eval_steps_per_second'])
    logger.log("epoch", metrics['epoch'])
    logger.log("trainerlog", trainer.state.log_history)
