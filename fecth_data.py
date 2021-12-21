import os
import pandas as pd
import valohai
import wandb

wandb.login()
wandb.init(project="huggin-project", entity="eikku")

inputs = {"attacks": "datum://017c9ddc-ce57-672e-9358-50a2ffb9115e"}
valohai.prepare(step="fetch_data", default_inputs=inputs)

file_path = valohai.inputs('attacks').path()
attacks = pd.read_csv(file_path,encoding='cp1252')

wandb.log_artifact(attacks, name='data_artifact', type='my_dataset')
wandb.log({"table": attacks})

attacks.columns = attacks.columns.str.strip()

df=attacks[["Case Number", "Type","Activity","Fatal (Y/N)","Injury",'Species']]
df=df[df["Species"].notnull()]
df=df[df["Injury"].notnull()]

path = valohai.outputs('*').path('attacksmini.csv')
df.to_csv(path)
