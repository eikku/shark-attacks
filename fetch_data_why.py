import os
import pandas as pd
import valohai

inputs = {"attacks": "datum://017c9ddc-ce57-672e-9358-50a2ffb9115e"}
valohai.prepare(step="fetch_data_why", default_inputs=inputs)

from whylogs.app import Session
from whylogs.app.writers import WhyLabsWriter

os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-gCqMEm"

# Adding the WhyLabs Writer to utilize WhyLabs platform
writer = WhyLabsWriter("", formats=[])

session = Session(project="demo-project", pipeline="demo-pipeline", writers=[writer])

# Run whylogs on current data and upload to WhyLabs.
with session.logger(tags={"datasetId": "model-1"}) as ylog:
  file_path = valohai.inputs('attacks').path()
  attacks = pd.read_csv(file_path,encoding='cp1252')
  ylog.log_dataframe(attacks)

attacks.columns = attacks.columns.str.strip()

df=attacks[["Case Number", "Type","Activity","Fatal (Y/N)","Injury",'Species']]
df=df[df["Species"].notnull()]
df=df[df["Injury"].notnull()]

path = valohai.outputs('*').path('attacksmini.csv')
df.to_csv(path)
