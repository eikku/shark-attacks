
import valohai
inputs = {"train_dataset":[],"val_dataset":[],"test_dataset":[]}
valohai.prepare(step="fine_tune", default_inputs=inputs)

file_path = valohai.inputs('train_dataset').path()

file_path = valohai.inputs('val_dataset').path()

file_path = valohai.inputs('test_dataset').path()


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


###Accuracy
predictions=trainer.predict(test_dataset)

test_results=test.copy(deep=True)
test_results["label_int_pred_transfer_learning"]=predictions.label_ids
test_results['label_pred_transfer_learning']=test_results['label_int_pred_transfer_learning'].apply(lambda x:labels[x])


##miten printtaa tulokset?
